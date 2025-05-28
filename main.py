from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from extract_frame import extract_first_frame

app = FastAPI()

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)
mp_hands = mp.solutions.hands.Hands(static_image_mode=True)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

@app.post("/pose")
async def detect_pose(file: UploadFile = File(...)):
    image_rgb = extract_first_frame(file)
    if image_rgb is None:
        return JSONResponse(content={"error": "Failed to extract frame."}, status_code=400)

    results = mp_pose.process(image_rgb)
    if not results.pose_landmarks:
        return JSONResponse(content={"error": "No pose landmarks found."}, status_code=400)

    landmarks = [
        {"x": lm.x, "y": lm.y, "z": lm.z}
        for lm in results.pose_landmarks.landmark
    ]
    return {"pose": landmarks}

@app.post("/hands")
async def detect_hands(file: UploadFile = File(...)):
    image_rgb = extract_first_frame(file)
    if image_rgb is None:
        return JSONResponse(content={"error": "Failed to extract frame."}, status_code=400)

    results = mp_hands.process(image_rgb)
    all_hands = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in hand_landmarks.landmark
            ]
            all_hands.append(landmarks)
    return {"hands": all_hands}

@app.post("/face")
async def detect_face(file: UploadFile = File(...)):
    image_rgb = extract_first_frame(file)
    if image_rgb is None:
        return JSONResponse(content={"error": "Failed to extract frame."}, status_code=400)

    results = mp_face.process(image_rgb)
    if not results.multi_face_landmarks:
        return JSONResponse(content={"error": "No face landmarks found."}, status_code=400)

    all_faces = []
    for face_landmarks in results.multi_face_landmarks:
        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in face_landmarks.landmark
        ]
        all_faces.append(landmarks)
    return {"face": all_faces}