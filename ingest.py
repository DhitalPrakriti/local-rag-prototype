import os 

def load_document(path):
    with open(path, "r" , encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=300,overlap=50):
    chunks =[]
    start = 0 
    while start < len(text): 
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk);
        start = end - overlap 
    return chunks
    
if __name__ == "__main__":
    doc_path = "data/vancouver_weather.txt"
    text = load_document(doc_path)
    chunks = chunk_text(text)
    print(f"Loaded document with {len(chunks)} chunks")
    print("\nSample chunk:\n")
    print(chunks[0])
    
        
        