import os
import faiss
import pickle

INDEX_FILE = "vector_index.index"
CHUNKS_FILE = "chunks.pkl"

def save_faiss_index(index):
    """Save FAISS index to a file."""
    faiss.write_index(index, INDEX_FILE)

def load_faiss_index(dimension):
    """Load FAISS index from a file or create a new one using cosine similarity."""
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)

    # ✅ Create FAISS index using cosine similarity if not found
    index = faiss.IndexFlatIP(dimension)
    return index


def save_chunks(chunks):
    """Save text chunks to a file."""
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    """Load document chunks from a file and ensure they match FAISS."""
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)

        # Check if FAISS exists and matches chunks
        if os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
            num_vectors = index.ntotal

            if len(chunks) != num_vectors:
                print(f"⚠️ FAISS and Chunks MISMATCHED! ({num_vectors} vectors vs {len(chunks)} chunks)")
                print("❌ Deleting and rebuilding FAISS index...")
                os.remove(INDEX_FILE)
                os.remove(CHUNKS_FILE)
                return []
        return chunks
    return []
