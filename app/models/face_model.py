import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
from pathlib import Path
from app.core.logger import setup_logger
from app.core.config import settings

logger = setup_logger(__name__)

class FaceModel:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = self._create_model() if not Path(settings.FACE_MODEL_PATH).exists() else self._load_model()
        
        # Load the face detection cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Error loading face cascade classifier")
            raise ValueError("Could not load face cascade classifier")
        logger.info("Face cascade classifier loaded successfully")
        
    def _create_model(self):
        """Create the CNN model architecture"""
        model = Sequential()

        # 1st CNN layer
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd CNN layer
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Fully connected 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected 3rd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.22))

        # Output layer
        model.add(Dense(len(self.emotions), activation='softmax'))

        # Compile model
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        logger.info("Created new face emotion recognition model")
        return model
        
    def _load_model(self):
        """Load the face emotion recognition model"""
        try:
            model = tf.keras.models.load_model(settings.FACE_MODEL_PATH)
            logger.info("Face model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading face model: {e}")
            raise
            
    def predict(self, img_array):
        """Predict emotion from face image"""
        try:
            # Convert image array to BGR format if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = img_array
            else:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Convert to grayscale for face detection
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                logger.warning("No faces detected in the image")
                return {"error": "No faces detected in the image"}

            # Find the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Crop and process the face
            processed_face = self._preprocess_image(gray_img[y:y+h, x:x+w])
            
            # Get predictions
            predictions = self.model.predict(processed_face)
            
            # Get all emotion probabilities
            emotion_probs = {
                emotion: float(prob) * 100 
                for emotion, prob in zip(self.emotions, predictions[0])
            }
            
            # Get the highest probability emotion
            emotion_index = predictions.argmax()
            emotion = self.emotions[emotion_index]
            confidence = float(predictions[0][emotion_index])
            
            # Return face location and predictions
            return {
                "emotion": emotion,
                "confidence": confidence * 100,  # Convert to percentage
                "face_location": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "all_emotions": emotion_probs
            }
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            raise
            
    def _preprocess_image(self, face_img):
        """Preprocess face image for model input"""
        try:
            # Resize to model input size (48x48)
            resized_face = cv2.resize(face_img, (48, 48))
            
            # Normalize pixel values
            normalized_face = resized_face
            
            # Reshape for model input (add batch and channel dimensions)
            processed_face = normalized_face.reshape(1, 48, 48, 1)
            
            return processed_face
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise