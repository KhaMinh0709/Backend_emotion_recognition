from fastapi import HTTPException, UploadFile
import cv2
import numpy as np
from app.core.config import settings
import aiofiles
from pathlib import Path

async def validate_image(file: UploadFile):
    """Validate uploaded image file"""
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
        )
    
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large. Maximum size: {settings.MAX_UPLOAD_SIZE/1024/1024}MB"
        )

async def save_upload_file(upload_file: UploadFile, folder: str) -> Path:
    """Save uploaded file and return the path"""
    try:
        file_path = Path(settings.UPLOAD_DIR) / folder / upload_file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
            
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

async def load_image_into_numpy_array(file: UploadFile):
    """Load image from UploadFile into numpy array"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def save_result_image(original_img: np.ndarray, face_location: dict, emotion: str, 
                     confidence: float, file_name: str) -> str:
    """Save result image with face detection box and prediction"""
    try:
        img_with_box = original_img.copy()
        x, y, w, h = face_location['x'], face_location['y'], face_location['width'], face_location['height']
        
        # Draw face rectangle
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add prediction text
        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(img_with_box, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
        
        # Save result
        result_path = Path(settings.RESULTS_DIR) / file_name
        result_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result_path), img_with_box)
        
        return str(result_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save result image: {str(e)}")