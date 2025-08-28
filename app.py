from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import tempfile
import os
import uvicorn

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # switch to custom model if trained

app = FastAPI(title="Home Objects Detection API")

def detect_and_annotate_image(img):
    # Run detection
    results = model.predict(source=img, save=False, conf=0.25, imgsz=640)

    detections = []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "class_id": int(cls),
                "class_name": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })

    # Annotate image
    annotated_img = results[0].plot()
    _, img_encoded = cv2.imencode(".jpg", annotated_img)
    return detections, BytesIO(img_encoded.tobytes())

@app.post("/detect/image/")
async def detect_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    detections, annotated_img_bytes = detect_and_annotate_image(img)

    return StreamingResponse(annotated_img_bytes, media_type="image/jpeg")

@app.post("/detect/video/")
async def detect_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(await file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.25, imgsz=640)
        annotated_frame = results[0].plot()

        if out is None:
            h, w = annotated_frame.shape[:2]
            out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (w, h))

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Return annotated video
    return StreamingResponse(open(temp_output.name, "rb"), media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
