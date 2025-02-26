from scraper import scrape_and_chunk
from faiss_manager import save_faiss_index, save_chunks, load_faiss_index, load_chunks
from gemini_rag import rag_pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import asyncio
import os

# Fix asyncio issue in Python 3.10+
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Fix Streamlit watching unnecessary files
os.environ["STREAMLIT_WATCH_FILE"] = "false"


dimension = 384
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load existing FAISS index and chunks
chunks = load_chunks()
index = load_faiss_index(dimension)

if not chunks or index.ntotal == 0:
    print("⚡ Creating new FAISS index...")
    urls = [
    "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
    "https://arxiv.org/list/cs.AI/recent",
    "https://openai.com/research/",
    "https://ai.googleblog.com/",
    "https://www.technologyreview.com/topic/artificial-intelligence/",
    "https://venturebeat.com/category/ai/",
     "https://huggingface.co/blog",
    "https://www.ibm.com/topics/artificial-intelligence",  # ✅ IBM AI Learning
    "https://aws.amazon.com/machine-learning/",  # ✅ AWS AI Guides
    "https://developers.google.com/machine-learning/",  # ✅ Google AI Developer Tutorials
    "https://www.coursera.org/browse/data-science/ai",  # ✅ AI Courses & Learning
]

    chunks = scrape_and_chunk(urls)

    # ✅ Ensure scraped content has multiple chunks before continuing
    if len(chunks) < 2:
        print("❌ Error: Scraped content is too small! Try a different source.")
        exit()

    # ✅ Generate embeddings and normalize them
    embeddings = model.encode(chunks)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # ✅ Normalize embeddings

    # ✅ Load or create FAISS index with cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # ✅ Ensure FAISS index receives multiple vectors
    if embeddings.shape[0] > 1:
        index.add(np.array(embeddings))  # ✅ Add normalized embeddings
        save_faiss_index(index)
        save_chunks(chunks)
        print(f"✅ FAISS index updated with {index.ntotal} vectors.")
    else:
        print("❌ Error: FAISS did not receive enough embeddings! Check chunking.")
        exit()
else:
    print(f"✅ FAISS index already exists with {index.ntotal} vectors.")

