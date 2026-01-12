import os
import csv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.retrieval import EquipmentMatcher
from src.scoring import calibrate_scores

app = FastAPI()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to src/
TEMPLATES_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")
FEEDBACK_FILE = os.path.join(BASE_DIR, "../manual_feedback.csv")

# --- Templates & static ---
templates = Jinja2Templates(directory=TEMPLATES_DIR)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Initialize Equipment Matcher ---
matcher = EquipmentMatcher(
    "data/processed/faiss_index.idx",
    "data/processed/faiss_index.idx_mapping.pkl"
)

# --- API endpoint for programmatic evaluation ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/evaluate")
def evaluate(request: QueryRequest):
    results = matcher.hybrid_search(request.query, request.top_k)
    results = calibrate_scores(results)
    return {"query": request.query, "results": results}

# --- Web UI for manual review ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": [], "query": ""}
    )

@app.post("/", response_class=HTMLResponse)
def manual_review(request: Request, query: str = Form(...), top_k: int = Form(5)):
    results = matcher.hybrid_search(query, top_k)
    results = calibrate_scores(results)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "query": query}
    )

# --- Feedback endpoint ---
@app.post("/feedback")
def feedback(query: str = Form(...), equipment_id: int = Form(...)):
    write_header = not os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["query", "equipment_id"])
        writer.writerow([query, equipment_id])
    return JSONResponse({"status": "saved", "query": query, "equipment_id": equipment_id})

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
