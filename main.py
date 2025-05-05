
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import uuid

app = FastAPI()
model = YOLO('yolov8n-pose.pt')

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, save_txt=False, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if keypoints.shape[1] >= 6:
            shoulder = keypoints[0][5]
            elbow = keypoints[0][7]
            hip = keypoints[0][11]
            angle = calculate_angle(shoulder, elbow, hip)
            cv2.putText(frame, f"Angulo: {angle:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)
            cv2.line(frame, tuple(hip.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return FileResponse("output.mp4", media_type="video/mp4", filename="processed_video.mp4")
  @app.get("/")
def read_root():
    return {"message": "EmboART Keralty API is running"}

