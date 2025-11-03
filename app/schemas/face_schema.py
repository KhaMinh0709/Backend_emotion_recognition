from pydantic import BaseModel
from typing import Dict

class FaceLocation(BaseModel):
    x: int
    y: int
    width: int
    height: int

class FaceResponse(BaseModel):
    emotion: str
    confidence: float
    face_location: FaceLocation
    all_emotions: Dict[str, float]

class FaceUploadResponse(BaseModel):
    message: str
    file_path: str