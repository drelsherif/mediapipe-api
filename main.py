from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()
mp_pose = mp.solutions.pose

@app.post("/pose")
async def detect_pose(file: UploadFile = File(...)):
    contents = await file.read()

    # Decode image
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return JSONResponse({"error": "No pose detected"}, status_code=404)

    landmarks = [
        {
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        }
        for lm in results.pose_landmarks.landmark
    ]

    return {"landmarks": landmarks}