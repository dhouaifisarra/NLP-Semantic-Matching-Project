import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EquipmentMatcher:
    def __init__(self, faiss_index_file: str, df_file: str):
        self.index = faiss.read_index(faiss_index_file)
        with open(df_file, "rb") as f:
            self.mapping = pickle.load(f)
        self.names = self.mapping['equipment_name']
        tokenized = [name.lower().split() for name in self.names]
        self.bm25 = BM25Okapi(tokenized)
        self.model = SentenceTransformer(MODEL_NAME)

    def semantic_search(self, query: str, top_k=5):
        emb = self.model.encode([query])
        D, I = self.index.search(emb, top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            results.append({
                "id": self.mapping['id'][i],
                "equipment_name": self.mapping['equipment_name'][i],
                "score": float(score)
            })
        return results

    def lexical_fallback(self, query: str, top_k=5):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{"id": self.mapping['id'][i], "equipment_name": self.mapping['equipment_name'][i], "score": float(scores[i])} for i in idxs]

    def hybrid_search(self, query: str, top_k=5):
        sem = self.semantic_search(query, top_k)
        lex = self.lexical_fallback(query, top_k)
        seen = set()
        combined = []
        for r in sem + lex:
            if r['id'] not in seen:
                combined.append(r)
                seen.add(r['id'])
        return combined[:top_k]
