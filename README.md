# NLP Semantic Matching Project: Equipment Matcher

This project implements an intelligent **Equipment Matcher** designed to resolve equipment descriptions using **Semantic Search** (embeddings), **Lexical Fallback**, and **Hybrid Search** strategies.

It provides both a **REST API** for programmatic evaluation and a **Web UI** for manual review and feedback collection.

---

## Project Structure
```text
nlp-semantic-matching-project/
├── data/
│   ├── raw/
│   │   └── equipment_catalog.csv         # Original catalog data
│   ├── processed/
│   │   ├── equipment_clean.csv           # Cleaned data
│   │   ├── faiss_index.idx               # Vector index (Generated)
│   │   └── faiss_index.idx_mapping.pkl   # Index mapping (Generated)
│   └── queries/
│       └── test_queries.csv              # Data for evaluation
├── src/
│   ├── api.py                            # FastAPI application entry point
│   ├── embeddings.py                     # Embedding generation logic
│   ├── retrieval.py                      # Search & retrieval logic (FAISS)
│   └── scoring.py                        # Scoring & ranking logic
├── templates/
│   └── index.html                        # Web UI HTML
├── static/
│   └── style.css                         # Web UI Styles
├── notebooks/
│   └── evaluation.ipynb                  # Metrics calculation notebook
├── manual_feedback.csv                   # Stores user feedback
├── requirements.txt                      # Python dependencies
└── Dockerfile                            # Container configuration
```

## Requirements
Python: 3.11

PyTorch: (CPU version)

FAISS: (CPU version)

Other dependencies listed in requirements.txt

## Setup & Installation

### 1. Clone the Repository
```bash 
git clone https://github.com/dhouaifisarra/NLP-Semantic-Matching-Project
cd NLP-Semantic-Matching-Project
```

### 2. Create Virtual Environment & Install Dependencies

#### Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Embeddings & Index
Before running the app, generate the FAISS vector index:
```bash
python src/embeddings.py
```

This generates faiss_index.idx and faiss_index.idx_mapping.pkl in data/processed/.

## Running the Application
### Start the Server
```bash
uvicorn src.api:app --reload
```
API Base URL: http://127.0.0.1:8000
Web UI: http://127.0.0.1:8000/

## API Usage
#### Programmatic Evaluation
**Endpoint:** POST /evaluate

**Request:**

JSON
```json
{
  "query": "hydraulic pump",
  "top_k": 5
}
```
(Note: top_k is optional, defaults to 5)

**Response:**

JSON
```json
{
  "query": "hydraulic pump",
  "results": [
    {
      "id": 101,
      "equipment_name": "Industrial Water Pump",
      "score": 0.674,
      "confidence": 0.614
    },
    {
      "id": 102,
      "equipment_name": "Electric Submersible Water Pump",
      "score": 0.902,
      "confidence": 0.571
    }
  ]
}
```
### Manual Review & Feedback (Web UI)
Open http://127.0.0.1:8000/.

1. Enter a query and choose Top K.
2. Click Search.
3. Mark correct matches using the "Mark as correct" buttons.
4. Feedback is saved to manual_feedback.csv.

**Feedback Endpoint** : POST /feedback

## Evaluation Metrics
Run the notebook to calculate Recall@1, Recall@5, and MRR.

**1. Launch Jupyter:**
```bash
jupyter notebook notebooks/evaluation.ipynb
```
The notebook reads test_queries.csv, queries the API, and outputs metrics:

Recall@1: 0.73
Recall@5: 0.94
MRR: 0.81

## Docker: Build & Run
**1. Build Image**
```bash
docker build -t nlp-matcher .
```

**2. Run Container**
```bash
docker run -p 8000:8000 nlp-matcher
```

**3. Access App**

Web UI: http://127.0.0.1:8000/

API: http://127.0.0.1:8000/evaluate