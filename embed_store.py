import pickle
import faiss
from sentence_transformers import SentenceTransformer
from ingest import load_document, chunk_text


# Configuration
DATA_PATH = "data/vancouver_weather.txt"
INDEX_PATH = "vector.index"
CHUNKS_PATH = "chunks.pkl"

def main():
    #Load and chunk the document
    text = load_document(DATA_PATH)
    chunks = chunk_text(text)
    
    # Load embedding model 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    #Create FAISS index
    dimension = embeddings.shape[1]  #Gives the length of one embedding vector 
    index =faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    #Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)  
        
    print(f"Stored {len(chunks)} chunks in vector database")
    
if __name__ == "__main__":
    main()
    
    
    
    
    

