"""
FastAPI backend for image quality assessment.

Endpoints:
  POST /api/analyze        - upload image, get quality analysis
  GET  /api/analyses       - list past results
  GET  /api/analyses/{id}  - get one result
  GET  /api/health         - status check
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from ml.inference import get_analyzer
from backend.database import init_db, save_analysis, get_analysis, get_all_analyses, get_analysis_count
from backend.schemas import AnalysisResponse, HealthResponse

app = FastAPI(title="Image Quality Assessment API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()


@app.on_event("startup")
async def startup():
    init_db()
    try:
        get_analyzer()
        print("Models loaded")
    except Exception as e:
        print(f"Models not loaded: {e} (run training notebooks first)")


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_image(file: UploadFile = File(...)):
    # validate extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(422, f"Unsupported file type '{ext}'")

    image_bytes = file.file.read()
    if len(image_bytes) == 0:
        raise HTTPException(422, "Empty file")
    if len(image_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(422, f"File too large (max {MAX_UPLOAD_SIZE_MB}MB)")

    try:
        result = get_analyzer().analyze(image_bytes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    filename = file.filename or "unknown"
    result["id"] = save_analysis(filename, result)
    result["filename"] = filename
    return result


@app.get("/api/analyses")
async def list_analyses(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {
        "analyses": get_all_analyses(limit=limit, offset=offset),
        "total": get_analysis_count(),
        "limit": limit, "offset": offset,
    }


@app.get("/api/analyses/{analysis_id}")
async def get_analysis_by_id(analysis_id: int):
    result = get_analysis(analysis_id)
    if not result:
        raise HTTPException(404, f"Analysis {analysis_id} not found")
    return result


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    try:
        analyzer = get_analyzer()
        loaded, mtype = True, analyzer.model_type
    except Exception:
        loaded, mtype = False, "none"

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded, model_type=mtype,
        total_analyses=get_analysis_count(),
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
