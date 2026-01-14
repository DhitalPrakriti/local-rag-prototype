import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
INDEX_PATH = "vector.index"
CHUNKS_PATH = "chunks.pkl"
TOP_K = 2  # Number of relevant passages to retrieve


def main():
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load stored chunks
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get user query
    query = input("Ask a question: ")

    # Embed query
    query_embedding = model.encode([query])

    # Perform similarity search
    distances, indices = index.search(query_embedding, TOP_K)

    print("\nRetrieved passages:\n")
    for i in indices[0]:
        print("-" * 40)
        print(chunks[i])


if __name__ == "__main__":
    main()
