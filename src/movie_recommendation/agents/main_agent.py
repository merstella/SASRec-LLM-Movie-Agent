import json
import os
import re

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from langchain.agents import create_agent as create_modern_agent
except ImportError:
    create_modern_agent = None

try:
    from langchain.tools import tool
except ImportError:
    from langchain_core.tools import tool

try:
    from langchain_groq import ChatGroq
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'langchain-groq'. Install with: pip install langchain-groq"
    ) from exc

from . import agent_utils

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


@tool
def get_user_movie_history(user_id: int):
    """Return recent watched movies with genres for the user."""
    return agent_utils.get_user_history(user_id)


@tool
def get_recommendations_and_rerank(user_id: int, query: str = ""):
    """Generate SASRec candidates, rerank by query, and return top item details."""
    candidate_ids = agent_utils.recommend_next_candidates(user_id, k=50)
    top_10_ids = agent_utils.rerank_with_query(candidate_ids, query, top_n=10)
    return agent_utils.get_item_details(top_10_ids)


tools = [get_user_movie_history, get_recommendations_and_rerank]
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

SYSTEM_PROMPT = """You are a professional AI Movie Recommendation Assistant.
Goal: recommend movies and produce high-quality reasons that feel personal and specific.

Mandatory workflow:
1. Call 'get_user_movie_history' first to infer preference patterns.
2. Call 'get_recommendations_and_rerank' with the same user_id and query.
3. Build final recommendations only from tool outputs.

Tool schema reminder:
- get_user_movie_history -> list of {{title, genres}}
- get_recommendations_and_rerank -> list of {{itemId, title, genres}}

Reason-writing rubric (apply to every recommendation):
- Write exactly 1 sentence, around 18-32 words.
- Mention at least one genre from that movie's 'genres'.
- Never claim genre/theme attributes not present in the movie's provided genres.
- Do not mention "space", "sci-fi", or similar unless the movie genres include Sci-Fi.
- Mention either:
  a) a key intent from the user's query, or
  b) an observed preference pattern from user history.
- Explain *why it fits* (tone, theme, pacing, or story vibe), not generic praise.
- Use varied phrasing across items; do not repeat the same opening pattern.
- Do not hallucinate unavailable facts (actors, director, year, ratings, plot details not provided by tools).

Output contract:
Return valid JSON only, no markdown, no extra commentary:
{{
  "recommendations": [
    {{"item_id": 123, "title": "Movie title", "reason": "One specific, grounded sentence"}}
  ]
}}

Hard constraints:
- Map 'itemId' from tool output to 'item_id' in final JSON.
- Preserve recommendation order from the ranking tool.
- If data is missing, still return valid JSON with best grounded reasons."""


if create_modern_agent is not None:
    agent_mode = "modern"
    agent = create_modern_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    agent_executor = None
else:
    agent_mode = "legacy"
    try:
        from langchain.agents import create_tool_calling_agent
    except ImportError:
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
    try:
        from langchain.agents import AgentExecutor
    except ImportError:
        from langchain.agents.agent_executor import AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def extract_output_text(result: dict) -> str:
    if agent_mode == "legacy":
        return str(result.get("output", ""))

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        msg_type = getattr(msg, "type", "")
        if msg_type != "ai":
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    return json.dumps(result, ensure_ascii=False, default=str)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _balance_json_closers(text: str) -> str:
    stack = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()

    if not stack:
        return text
    return text + "".join(reversed(stack))


def _repair_json_candidate(text: str):
    candidates = []
    base = text.strip()
    if not base:
        return candidates

    candidates.append(base)

    if len(base) >= 2 and base[0] == '"' and base[-1] == '"':
        try:
            unwrapped = json.loads(base)
            if isinstance(unwrapped, str):
                candidates.append(unwrapped.strip())
        except Exception:
            pass

    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", base)
    if no_trailing_commas not in candidates:
        candidates.append(no_trailing_commas)

    balanced = _balance_json_closers(no_trailing_commas)
    if balanced not in candidates:
        candidates.append(balanced)

    extracted = _extract_first_json_block(base)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
        extracted_balanced = _balance_json_closers(extracted)
        if extracted_balanced not in candidates:
            candidates.append(extracted_balanced)

    return candidates


def _extract_first_json_block(text: str):
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not starts:
        return None

    start = min(starts)
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
            continue
        if ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _normalize_recommendations_payload(data):
    if isinstance(data, list):
        data = {"recommendations": data}

    if not isinstance(data, dict):
        return None

    recs = data.get("recommendations")
    if not isinstance(recs, list):
        return None

    normalized = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        item_id = rec.get("item_id", rec.get("itemId"))
        title = rec.get("title")
        reason = rec.get("reason", "")
        if item_id is None or title is None:
            continue
        normalized.append(
            {
                "item_id": item_id,
                "title": str(title),
                "reason": str(reason),
            }
        )

    return {"recommendations": normalized}


def _extract_recommendations_regex(text: str):
    pattern = re.compile(
        r'\{\s*"(?P<id_key>item_id|itemId)"\s*:\s*(?P<item_id>\d+)\s*,\s*'
        r'"title"\s*:\s*"(?P<title>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"reason"\s*:\s*"(?P<reason>(?:\\.|[^"\\])*)"\s*\}'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    recommendations = []
    for m in matches:
        try:
            title = json.loads(f"\"{m.group('title')}\"")
            reason = json.loads(f"\"{m.group('reason')}\"")
        except Exception:
            title = m.group("title")
            reason = m.group("reason")

        recommendations.append(
            {
                "item_id": int(m.group("item_id")),
                "title": str(title),
                "reason": str(reason),
            }
        )

    return {"recommendations": recommendations}


def _extract_recommendations_partial(text: str):
    object_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    id_pattern = re.compile(r'"(?:item_id|itemId)"\s*:\s*(\d+)')
    title_pattern = re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"')
    reason_pattern = re.compile(r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"')

    recommendations = []
    for chunk_match in object_pattern.finditer(text):
        chunk = chunk_match.group(0)

        id_match = id_pattern.search(chunk)
        title_match = title_pattern.search(chunk)
        if not id_match or not title_match:
            continue

        item_id = int(id_match.group(1))
        try:
            title = json.loads(f"\"{title_match.group(1)}\"")
        except Exception:
            title = title_match.group(1)

        reason = ""
        reason_match = reason_pattern.search(chunk)
        if reason_match:
            try:
                reason = json.loads(f"\"{reason_match.group(1)}\"")
            except Exception:
                reason = reason_match.group(1)
        else:
            tail = chunk[title_match.end() :]
            tail = tail.replace("}", "").replace("\n", " ").strip(" ,\"'")
            tail = re.sub(r"^\W+", "", tail)
            if tail:
                reason = tail

        if not reason:
            reason = "Recommended based on your query and recent preference pattern."

        recommendations.append(
            {
                "item_id": item_id,
                "title": str(title),
                "reason": str(reason),
            }
        )

    if not recommendations:
        return None
    return {"recommendations": recommendations}


def _parse_llm_output(raw_text: str):
    base_variants = [raw_text, _strip_code_fences(raw_text)]
    candidates = []
    for variant in base_variants:
        for repaired in _repair_json_candidate(variant):
            if repaired and repaired not in candidates:
                candidates.append(repaired)

    parse_errors = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            parse_errors.append(f"{type(exc).__name__}: {exc}")
            continue

        normalized = _normalize_recommendations_payload(parsed)
        if normalized is not None:
            return normalized

    regex_fallback = _extract_recommendations_regex(raw_text)
    if regex_fallback is not None:
        return regex_fallback

    partial_fallback = _extract_recommendations_partial(raw_text)
    if partial_fallback is not None:
        return partial_fallback

    return {
        "recommendations": [],
        "raw_output": raw_text,
        "error": "LLM output was not valid/expected JSON.",
        "parse_errors": parse_errors[-2:],
    }


def invoke_agent(user_id: int, user_query: str):
    input_text = f"User ID: {user_id}. Query: {user_query}"
    if agent_mode == "modern":
        result = agent.invoke({"messages": [{"role": "user", "content": input_text}]})
    else:
        result = agent_executor.invoke({"input": input_text})
    text = extract_output_text(result)
    return _parse_llm_output(text)


if __name__ == "__main__":
    user_id = 1
    user_query = "I'm looking for action movies with science fiction or space elements."

    print(f"--- Processing {user_id} ---")
    print(f"--- Groq model: {GROQ_MODEL} ---")

    response = invoke_agent(user_id, user_query)
    print("\n--- AGENT RESULT ---")
    print(json.dumps(response, ensure_ascii=False, indent=2))
