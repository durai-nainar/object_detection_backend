from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

# Create FastAPI app
app = FastAPI()

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change later for security)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("yolov8n.pt")

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(img)

    # Draw detection results on image
    annotated = results[0].plot()

    # Encode image to JPEG for sending back
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
