from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from app.services.face_service import FaceService
from app.schemas.face_schema import FaceResponse, FaceUploadResponse
from typing import Dict, Any
import base64
import numpy as np
import cv2

router = APIRouter()
# Lazily create FaceService to avoid heavy model load at import/startup
face_service = None

def get_face_service() -> FaceService:
    global face_service
    if face_service is None:
        face_service = FaceService()
    return face_service

@router.post("/upload", response_model=FaceUploadResponse)
async def upload_face(file: UploadFile = File(...)):
    """Upload and process face image"""
    svc = get_face_service()
    result = await svc.process_image(file)
    return result

@router.post("/predict")
async def predict_emotion(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict emotion from face image.
    Returns:
    - emotion: Predicted emotion
    - confidence: Confidence score (percentage)
    - face_location: Bounding box coordinates of detected face
    - all_emotions: Probability scores for all emotions
    - result_image: Path to the result image with face detection box
    """
    svc = get_face_service()
    result = await svc.predict_emotion(file)
    return JSONResponse(content=result)

@router.post("/predict-webcam")
async def predict_webcam(request: Request):
    """
    Predict emotion from webcam image (base64)
    """
    try:
        data = await request.json()
        image_base64 = data.get("image_base64", "")

        # Convert base64 to numpy array
        b64_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Get prediction
        svc = get_face_service()
        result = await svc.predict_emotion(img)

        # Include face location for drawing on canvas
        return JSONResponse(content={
            "label": result["emotion"],
            "confidence": float(result["confidence"]),
            "face_location": result["face_location"],
            "all_emotions": result["all_emotions"]
        })
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )