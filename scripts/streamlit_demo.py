from pathlib import Path
import json
import os
import re
import sys
from urllib import parse, request

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommendation.agents import agent_utils
from movie_recommendation.agents.main_agent import GROQ_MODEL, invoke_agent

USERS_DAT_PATH = ROOT / "ml-1m" / "users.dat"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
AGE_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
OCCUPATION_LABELS = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


@st.cache_data(show_spinner=False)
def load_users_df(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["userId", "gender", "age", "occupation", "zip_code"])

    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        encoding="ISO-8859-1",
        dtype={"userId": int, "gender": str, "age": int, "occupation": int, "zip_code": str},
    )


def get_user_profile(user_id: int, users_df: pd.DataFrame):
    if users_df.empty:
        return None
    row = users_df.loc[users_df["userId"] == user_id]
    if row.empty:
        return None

    rec = row.iloc[0]
    gender_raw = str(rec["gender"]).upper()
    if gender_raw == "M":
        gender = "Male"
    elif gender_raw == "F":
        gender = "Female"
    else:
        gender = gender_raw

    age_code = int(rec["age"])
    occupation_code = int(rec["occupation"])
    return {
        "user_id": int(rec["userId"]),
        "gender": gender,
        "age_code": age_code,
        "age_group": AGE_LABELS.get(age_code, str(age_code)),
        "occupation_code": occupation_code,
        "occupation": OCCUPATION_LABELS.get(occupation_code, f"code {occupation_code}"),
        "zip_code": str(rec["zip_code"]),
    }


def get_user_history(user_id: int, n: int):
    history = agent_utils.get_user_history(user_id=user_id, n=n)
    if not isinstance(history, list):
        return []
    return history


def summarize_top_genres(history):
    counts = {}
    for row in history:
        genres = str(row.get("genres", ""))
        for g in genres.split("|"):
            g = g.strip()
            if not g:
                continue
            counts[g] = counts.get(g, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [g for g, _ in ranked[:5]]


def split_title_year(title: str):
    match = re.match(r"^(.*?)(?:\s*\((\d{4})\))?$", title.strip())
    if not match:
        return title.strip(), None
    name = match.group(1).strip()
    year = match.group(2)
    return name, int(year) if year else None


def _tmdb_request(params: dict):
    query_string = parse.urlencode(params)
    with request.urlopen(f"{TMDB_SEARCH_URL}?{query_string}", timeout=8) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _pick_poster_path(results, year):
    if year is not None:
        for movie in results:
            release = str(movie.get("release_date", ""))
            if release[:4] == str(year) and movie.get("poster_path"):
                return movie["poster_path"]
    for movie in results:
        if movie.get("poster_path"):
            return movie["poster_path"]
    return None


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(title: str, tmdb_api_key: str):
    if not tmdb_api_key:
        return None

    name, year = split_title_year(title)
    params = {
        "api_key": tmdb_api_key,
        "query": name,
        "include_adult": "false",
    }
    if year is not None:
        params["year"] = year

    try:
        data = _tmdb_request(params)
        results = data.get("results", [])
        poster_path = _pick_poster_path(results, year)
        if poster_path:
            return f"{TMDB_IMAGE_BASE}{poster_path}"

        if year is not None:
            params.pop("year", None)
            data = _tmdb_request(params)
            results = data.get("results", [])
            poster_path = _pick_poster_path(results, None)
            if poster_path:
                return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return None
    return None


st.set_page_config(page_title="Movie Recommendation Demo", layout="wide")
st.title("Movie Recommendation Demo")
st.caption(f"Model: {GROQ_MODEL}")

users_df = load_users_df(str(USERS_DAT_PATH))
tmdb_api_key = os.getenv("TMDB_API_KEY", "").strip()

col1, col2 = st.columns([1, 2])
with col1:
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    profile = get_user_profile(int(user_id), users_df)
    st.subheader("User Profile")
    if profile is None:
        if users_df.empty:
            st.warning("`ml-1m/users.dat` not found.")
        else:
            st.warning("User ID not found in `users.dat`.")
    else:
        p1, p2 = st.columns(2)
        p1.metric("Gender", profile["gender"])
        p2.metric("Age Group", profile["age_group"])
        st.caption(f"Occupation: {profile['occupation']} (code {profile['occupation_code']})")
        st.caption(f"ZIP Code: {profile['zip_code']}")

    query = st.text_area(
        "Query",
        value="I'm looking for action movies with science fiction or space elements.",
        height=120,
    )
    history_n = st.slider("Watched history length", min_value=5, max_value=30, value=12, step=1)
    watched_history = get_user_history(int(user_id), history_n)
    top_genres = summarize_top_genres(watched_history)
    if not tmdb_api_key:
        st.info("Add `TMDB_API_KEY` to `.env` to show movie posters.")
    submitted = st.button("Recommend", type="primary", use_container_width=True)

# Sidebar context panel
st.sidebar.header("Watched History")
if watched_history:
    if top_genres:
        st.sidebar.caption("Top genres: " + ", ".join(top_genres))
    for idx, row in enumerate(watched_history, start=1):
        title = row.get("title", "Unknown")
        genres = row.get("genres", "")
        st.sidebar.markdown(f"**{idx}. {title}**")
        if genres:
            st.sidebar.caption(genres)
else:
    st.sidebar.info("No watch history found for this user.")

with col2:
    st.subheader("User Context")
    if watched_history:
        context_df = pd.DataFrame(watched_history)
        st.dataframe(context_df, use_container_width=True, hide_index=True)
    else:
        st.info("No watched history to display.")

    if submitted:
        with st.spinner("Generating recommendations..."):
            result = invoke_agent(int(user_id), query)

        recs = result.get("recommendations", [])
        if not recs:
            st.warning("No recommendations returned.")
        else:
            st.subheader("Recommended Movies")
            st.caption("Poster source: TMDB")
            cols = st.columns(3)
            for idx, rec in enumerate(recs, start=1):
                title = rec.get("title", "Unknown")
                item_id = rec.get("item_id", "N/A")
                reason = rec.get("reason", "")
                poster_url = get_poster_url(str(title), tmdb_api_key)

                with cols[(idx - 1) % 3]:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown("`No poster`")
                    st.markdown(f"**{idx}. {title}**")
                    st.caption(f"item_id={item_id}")
                    st.write(reason)
                    st.divider()

        if result.get("error") or result.get("raw_output"):
            with st.expander("Debug Output"):
                st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
    else:
        st.info("Enter a user ID and query, then click Recommend.")
