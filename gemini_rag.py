import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from faiss_manager import load_faiss_index, load_chunks
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Gemini API key not found! Please check your .env file.")


# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
print("✅ Gemini API key loaded successfully!")

model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384

def rag_pipeline(user_query):
    """Retrieve relevant text from FAISS and generate a response using Gemini Pro."""
    index = load_faiss_index(dimension)
    chunks = load_chunks()

    query_embedding = model.encode([user_query])
    k = 5  # Retrieve top 5 similar chunks
    distances, indices = index.search(np.array(query_embedding), k)

    print(f"🔍 FAISS Retrieval Results:\nDistances: {distances}\nIndices: {indices}")

    # ✅ Increase similarity threshold to filter weak matches
    threshold = 0.85  # Adjust to get only highly relevant results
    relevant_chunks = [chunks[i] for i, d in zip(indices[0], distances[0]) if d < threshold]

    if not relevant_chunks:
        return "⚠️ No relevant chunks found. Try modifying your query or re-scraping."

    # ✅ Improve context formatting for better understanding
    context = "\n\n".join(relevant_chunks)
    prompt = f"### Context:\n{context}\n\n### Question:\n{user_query}\n\n### Answer (Explain in detail and provide examples):"

    gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = gemini_model.generate_content(prompt, stream=False)

    return f"📖 **Context Extracted:**\n```\n{context}\n```\n\n🧠 **AI Response:**\n**{response.text}**"
