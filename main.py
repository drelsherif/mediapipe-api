from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

@app.post("/pose")
async def detect_pose(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return JSONResponse({"error": "No pose detected"}, status_code=404)

        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]
        return {"landmarks": landmarks}

@app.post("/hands")
async def detect_hands(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return JSONResponse({"error": "No hands detected"}, status_code=404)

        all_hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
            all_hands.append(hand)
        return {"hands": all_hands}

@app.post("/face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face:
        results = face.process(image_rgb)
        if not results.multi_face_landmarks:
            return JSONResponse({"error": "No face detected"}, status_code=404)

        face_landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in results.multi_face_landmarks[0].landmark
        ]
        return {"face": face_landmarks}