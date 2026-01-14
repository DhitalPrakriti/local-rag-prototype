import pickle
import faiss
import subprocess
from sentence_transformers import SentenceTransformer

# Configuration
INDEX_PATH = "vector.index"
CHUNKS_PATH = "chunks.pkl"
TOP_K = 2
OLLAMA_MODEL = "llama3.2:1b"


def retrieve_context(query, model, index, chunks):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, TOP_K)

    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    return retrieved_chunks


def build_prompt(context_chunks, question):
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a question-answering system.

RULES:
- Answer ONLY using facts that appear in the context.
- Extract the relevant sentence(s) from the context and rewrite them as a direct answer.
- Do NOT add new information.
- If the answer cannot be found in the context, say:
  "I don't know based on the provided documents."

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()


def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )
    return result.stdout.strip()


def main():
    # Load vector index and chunks
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get user question
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Retrieve context
        context_chunks = retrieve_context(question, model, index, chunks)

        # Build grounded prompt
        prompt = build_prompt(context_chunks, question)

        # Call local LLM
        answer = call_ollama(prompt)

        # Display results
        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"[{i}] {chunk[:200]}...")
    
if __name__ == "__main__":
    main()
