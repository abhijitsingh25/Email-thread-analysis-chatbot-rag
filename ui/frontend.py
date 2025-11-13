# ui/streamlit_app.py
import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Email RAG", layout="wide")
st.title("📧 Email + Attachment RAG")

# ------------------------------
# Load Thread Labels
# ------------------------------
with open("data/thread_map.json") as f:
    thread_map = json.load(f)

# T-000X list
thread_labels = sorted(thread_map.values())

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ------------------------------
# Thread selector (DROPDOWN)
# ------------------------------
thread_label = st.selectbox("Select Thread:", thread_labels)

# ------------------------------
# Start Session Button
# ------------------------------
if st.button("Start Session"):
    # send thread_label instead of thread_id
    r = requests.post(f"{API_URL}/start_session", json={"thread_label": thread_label})

    if r.ok:
        sid = r.json()["session_id"]
        st.session_state.session_id = sid
        st.success(f"Session started: {sid}")
    else:
        st.error("Failed to start session")

# ------------------------------
# Question UI
# ------------------------------
if st.session_state.session_id:
    q = st.text_area("Ask a question:")
    if st.button("Ask"):
        payload = {
            "session_id": st.session_state.session_id,
            "question": q
        }
        r = requests.post(f"{API_URL}/ask", json=payload)

        if r.ok:
            res = r.json()
            st.subheader("Answer:")
            st.write(res["answer"])
            st.markdown("**Citations:**")
            for c in res["citations"]:
                st.write(f"- {c}")
        else:
            st.error("Error fetching answer")
