import os
import cv2
import json
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from mediapipe.python.solutions import pose as mp_pose

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose = mp_pose.Pose(static_image_mode=False)

@app.post("/pose")
async def pose_from_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic"]:
        return {"error": "Unsupported image format."}

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image."}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            return {"error": "No pose landmarks found."}

        return {
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/pose_video")
async def pose_from_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi"]:
        return {"error": "Unsupported video format."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        return {"error": "Failed to open video."}

    results = []
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = pose.process(frame_rgb)
        if mp_result.pose_landmarks:
            results.append({
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in mp_result.pose_landmarks.landmark
                ]
            })

    cap.release()
    os.remove(tmp_path)

    if not results:
        return {"error": "No pose landmarks found in video."}

    return {"frames": results}
