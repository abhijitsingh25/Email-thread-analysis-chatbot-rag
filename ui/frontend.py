# streamlit_app.py
import streamlit as st
import requests, time

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Email Thread RAG", layout="wide")
st.title("Email + Attachment RAG (thread-focused)")

# left: controls
with st.sidebar:
    st.header("Session")
    thread_id = st.text_input("Thread ID (e.g., T-0042)", value="")
    if st.button("Start session"):
        r = requests.post(f"{API_URL}/start_session", json={"thread_id": thread_id})
        if r.ok:
            sess = r.json()
            st.session_state.session_id = sess["session_id"]
            st.success(f"Session started: {sess['session_id']} (thread {thread_id})")
    if st.button("Reset session"):
        requests.post(f"{API_URL}/reset_session")
        st.session_state.clear()
        st.experimental_rerun()

    st.write("Toggle search outside thread")
    search_outside = st.checkbox("Search outside thread (fallback)")

# main chat area
if "session_id" not in st.session_state:
    st.info("Start a session first from the sidebar.")
else:
    st.header("Chat")
    user_input = st.text_input("Ask a question about the selected thread")
    if st.button("Send"):
        payload = {"session_id": st.session_state.session_id, "text": user_input, "search_outside_thread": search_outside}
        r = requests.post(f"{API_URL}/ask", json=payload)
        if r.ok:
            res = r.json()
            st.subheader("Answer")
            st.write(res["answer"])
            st.markdown("**Citations**")
            for c in res["citations"]:
                st.write(c)
            st.expander("Debug: rewrite & retrieved", expanded=False)
            with st.expander("Debug"):
                st.write("Rewrite:", res["rewrite"])
                st.write("Top retrieved (ids & scores):")
                for it in res["retrieved"]:
                    meta = it["meta"]
                    st.write(f"{meta.get('message_id')} | score: {it['score']} | source: {meta.get('source')} | page: {meta.get('page_no')}")
