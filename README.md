# Backend Emotion Recognition API

This project implements a FastAPI-based backend service for emotion recognition using both facial and audio inputs.

## Project Structure

```
BACKEND_EMOTION_RECOGNITION/
│
├── app/                         # Main application package
│   ├── main.py                 # FastAPI application entry point
│   ├── api/                    # API route definitions
│   ├── core/                   # Core configuration
│   ├── models/                 # ML/DL model implementations
│   ├── services/               # Business logic layer
│   ├── schemas/               # Pydantic models
│   ├── utils/                 # Utility functions
│   └── static/                # Static file storage
│
├── models/                    # Trained model files
└── requirements.txt          # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained models in the `models/` directory:
- `face_model.h5` for facial emotion recognition
- `audio_model.pth` for audio emotion recognition
- `fusion_model.pth` for multimodal fusion

## Running the Application

Start the server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Face Emotion Recognition
- POST `/face/upload`: Upload face image
- POST `/face/predict`: Predict emotion from face image

### Audio Emotion Recognition
- POST `/audio/upload`: Upload audio file
- POST `/audio/predict`: Predict emotion from audio

### Multimodal Fusion
- POST `/fusion/predict`: Predict emotion using both face and audio inputs

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
DEBUG=True
MAX_UPLOAD_SIZE=10485760
```