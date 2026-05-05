"""FastAPI app: serves the frontend and exposes /api/extract."""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kifu_extractor import extract_kifu


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Kifu Extractor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/extract")
async def api_extract(
    image: UploadFile = File(...),
    debug: str = Form("false"),
):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(400, "uploaded file must be an image")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        want_debug = debug.lower() in {"1", "true", "yes", "on"}
        result = extract_kifu(data, return_debug=want_debug)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"extraction failed: {exc}") from exc
    return JSONResponse(result)


# Serve the static frontend if present.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
