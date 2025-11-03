from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.api import face_routes

app = FastAPI(title="Emotion Recognition API")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(face_routes.router, prefix="/face", tags=["Face"])
# app.include_router(webcam_routes.router, prefix="/webcam", tags=["Webcam"])
# app.include_router(audio_routes.router, prefix="/audio", tags=["Audio"])
# app.include_router(fusion_routes.router, prefix="/fusion", tags=["Fusion"])

@app.get("/")
async def root():
    return {"message": "Welcome to Emotion Recognition API"}

@app.get("/webcam", response_class=HTMLResponse)
async def webcam():
    """Serve the webcam interface"""
    with open("app/static/webcam.html", "r") as f:
        return f.read()


@app.get("/health")
async def health():
    """Health check endpoint to verify server is running"""
    return {"status": "ok"}

# start the server with:
# python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000