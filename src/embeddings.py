import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_embeddings(input_csv: str, index_file: str):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer(MODEL_NAME)
    
    embeddings = model.encode(df['clean_name'].tolist(), convert_to_numpy=True)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, index_file)
    
    # Save mapping for retrieval
    with open(index_file + "_mapping.pkl", "wb") as f:
        pickle.dump(df[['id', 'equipment_name']].to_dict(orient='list'), f)
    
    print(f"FAISS index saved to {index_file} and mapping saved.")

if __name__ == "__main__":
    create_embeddings("data/processed/equipment_clean.csv", "data/processed/faiss_index.idx")

