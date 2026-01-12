import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calibrate_scores(results):
    # Convert FAISS distance to similarity
    max_score = max([r['score'] for r in results]) + 1e-6
    for r in results:
        r['confidence'] = float(sigmoid(1 - r['score']/max_score))
    return results
