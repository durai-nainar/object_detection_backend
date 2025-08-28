from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os
import asyncio
import json
import base64
import tempfile
import time
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Performance settings
FRAME_SKIP = 2  # Process every 2nd frame for live detection
CONFIDENCE_THRESHOLD = 0.5

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

manager = ConnectionManager()

def resize_for_speed(image, max_size=640):
    """Resize image for faster processing while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

@app.get("/")
async def root():
    return {"message": "Object Detection API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_websockets": len(manager.active_connections)
    }

# Your existing image detection endpoint (unchanged)
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Run YOLO detection
    results = model(img, conf=CONFIDENCE_THRESHOLD)
    
    # Draw detection results on image
    annotated = results[0].plot()
    
    # Encode image to JPEG for sending back
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

# NEW: Video detection endpoint
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """Process uploaded video and return annotated version"""
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(await file.read())
            input_path = temp_input.name
        
        output_path = input_path.replace(".mp4", "_detected.mp4")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        logger.info(f"Processing video: {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame for speed, copy others
            if frame_count % FRAME_SKIP == 0:
                try:
                    # Resize for faster processing
                    resized_frame = resize_for_speed(frame)
                    
                    # Run detection
                    results = model(resized_frame, conf=CONFIDENCE_THRESHOLD)
                    annotated_frame = results[0].plot()
                    
                    # Resize back to original size
                    if annotated_frame.shape[:2] != (height, width):
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                    
                    out.write(annotated_frame)
                except Exception as e:
                    logger.warning(f"Frame {frame_count} processing failed: {e}")
                    out.write(frame)
            else:
                out.write(frame)
            
            # Log progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # Clean up input file
        os.unlink(input_path)
        
        # Return processed video
        def cleanup():
            try:
                os.unlink(output_path)
            except:
                pass
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="detected_video.mp4",
            background=cleanup
        )
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        # Cleanup
        if 'input_path' in locals() and os.path.exists(input_path):
            os.unlink(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(status_code=500, detail=str(e))

# NEW: WebSocket endpoint for live detection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time object detection via WebSocket"""
    await manager.connect(websocket)
    
    try:
        frame_count = 0
        last_time = time.time()
        
        while True:
            try:
                # Receive frame data
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "frame":
                    frame_count += 1
                    current_time = time.time()
                    
                    # Process every nth frame for performance
                    if frame_count % FRAME_SKIP == 0:
                        try:
                            # Decode base64 image
                            frame_data = message.get("data", "")
                            if frame_data.startswith("data:image"):
                                frame_data = frame_data.split(",")[1]
                            
                            # Convert to OpenCV image
                            img_bytes = base64.b64decode(frame_data)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Resize for speed
                                resized_frame = resize_for_speed(frame)
                                
                                # Run detection
                                results = model(resized_frame, conf=CONFIDENCE_THRESHOLD)
                                annotated_frame = results[0].plot()
                                
                                # Resize back if needed
                                if annotated_frame.shape != frame.shape:
                                    annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                                
                                # Encode back to base64
                                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                encoded_frame = base64.b64encode(buffer).decode()
                                
                                # Calculate FPS
                                fps = 1 / max(0.001, current_time - last_time)
                                last_time = current_time
                                
                                # Send result
                                response = {
                                    "type": "detection_result",
                                    "data": f"data:image/jpeg;base64,{encoded_frame}",
                                    "fps": round(fps, 1),
                                    "frame_count": frame_count
                                }
                                
                                await websocket.send_text(json.dumps(response))
                            
                        except Exception as e:
                            logger.error(f"Frame processing error: {e}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"Processing error: {str(e)}"
                            }))
                    else:
                        # Send skip acknowledgment
                        await websocket.send_text(json.dumps({
                            "type": "frame_skipped",
                            "frame_count": frame_count
                        }))
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(websocket)

# NEW: Configuration endpoint
@app.post("/configure")
async def configure_detection(
    confidence: float = None,
    frame_skip: int = None
):
    """Configure detection parameters"""
    global CONFIDENCE_THRESHOLD, FRAME_SKIP
    
    updated = {}
    
    if confidence is not None and 0.1 <= confidence <= 1.0:
        CONFIDENCE_THRESHOLD = confidence
        updated["confidence"] = confidence
    
    if frame_skip is not None and frame_skip >= 1:
        FRAME_SKIP = frame_skip
        updated["frame_skip"] = frame_skip
    
    return {
        "message": "Configuration updated",
        "updated": updated,
        "current_settings": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "frame_skip": FRAME_SKIP
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
