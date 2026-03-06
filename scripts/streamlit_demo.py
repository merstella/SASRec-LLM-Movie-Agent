from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommendation.agents.main_agent import GROQ_MODEL, invoke_agent

USERS_DAT_PATH = ROOT / "ml-1m" / "users.dat"
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


st.set_page_config(page_title="Movie Recommendation Demo", page_icon="🎬", layout="wide")
st.title("Movie Recommendation Demo")
st.caption(f"Model: {GROQ_MODEL}")

users_df = load_users_df(str(USERS_DAT_PATH))

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
    submitted = st.button("Recommend", type="primary", use_container_width=True)

with col2:
    if submitted:
        with st.spinner("Generating recommendations..."):
            result = invoke_agent(int(user_id), query)

        recs = result.get("recommendations", [])
        if not recs:
            st.warning("No recommendations returned.")
        else:
            st.subheader("Recommended Movies")
            for idx, rec in enumerate(recs, start=1):
                title = rec.get("title", "Unknown")
                item_id = rec.get("item_id", "N/A")
                reason = rec.get("reason", "")
                st.markdown(f"**{idx}. {title}** (`item_id={item_id}`)")
                st.write(reason)
                st.divider()

        if result.get("error") or result.get("raw_output"):
            with st.expander("Debug Output"):
                st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
    else:
        st.info("Enter a user ID and query, then click Recommend.")
