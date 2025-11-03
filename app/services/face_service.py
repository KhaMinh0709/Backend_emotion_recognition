from fastapi import UploadFile, HTTPException
from pathlib import Path
import numpy as np
# import aiofiles
from app.models.face_model import FaceModel
from app.utils.image_utils import (
    validate_image, 
    save_upload_file, 
    load_image_into_numpy_array,
    save_result_image
)
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

class FaceService:
    def __init__(self):
        self.model = FaceModel()
        
    async def process_image(self, file: UploadFile):
        """Process uploaded face image"""
        try:
            # Validate file
            await validate_image(file)
            
            # Save file
            file_path = await save_upload_file(file, "faces")
            
            return {"message": "File uploaded successfully", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=400, detail=str(e))
            
    async def predict_emotion(self, image_input):
        """Predict emotion from face image.
        Args:
            image_input: Either an UploadFile or a numpy array containing the image
        """
        try:
            if isinstance(image_input, np.ndarray):
                image_array = image_input
            else:
                # It's an UploadFile
                await validate_image(image_input)
                image_array = await load_image_into_numpy_array(image_input)
            
            # Get prediction
            result = self.model.predict(image_array)
            
            # Log the raw prediction result
            logger.info(f"Raw prediction result: {result}")
            
            if not isinstance(result, dict):
                logger.error(f"Unexpected result type: {type(result)}")
                return {"error": "Invalid prediction result format"}
            
            if "error" in result:
                logger.warning(f"Model returned error: {result['error']}")
                return {"error": result["error"]}
            
            # Validate required fields
            required_fields = ["emotion", "confidence", "face_location", "all_emotions"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                logger.error(f"Missing required fields in prediction result: {missing_fields}")
                return {"error": f"Invalid prediction result: missing {', '.join(missing_fields)}"}
            
            # Process the result to ensure all values are JSON serializable
            processed_result = {
                "emotion": result["emotion"],
                "confidence": float(result["confidence"]),
                "face_location": {
                    k: int(v) for k, v in result["face_location"].items()
                },
                "all_emotions": {
                    k: float(v) for k, v in result["all_emotions"].items()
                }
            }
            
            if not isinstance(image_input, np.ndarray):
                # Save result image only for uploaded files
                result_file = f"result_{image_input.filename}"
                result_path = save_result_image(
                    original_img=image_array,
                    face_location=processed_result["face_location"],
                    emotion=processed_result["emotion"],
                    confidence=processed_result["confidence"],
                    file_name=result_file
                )
                # Add result image path to response for uploaded files
                processed_result["result_image"] = str(result_path)
            
            return processed_result
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            raise HTTPException(status_code=400, detail=str(e))