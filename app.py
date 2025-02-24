import streamlit as st
from gemini_rag import rag_pipeline

st.title("🔍 AI-Powered Knowledge Search")

user_query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_query:
        answer = rag_pipeline(user_query)
        st.markdown(answer)
