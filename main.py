import os
import cv2
import json
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
from typing import List, Dict, Any, Optional
import subprocess

app = FastAPI(title="MediaPipe Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Create solution instances
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detection_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def get_video_rotation(video_path: str) -> int:
    """Extract rotation metadata from video file using ffprobe"""
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "json", video_path
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rotation_json = json.loads(result.stdout)
        
        if 'streams' in rotation_json and rotation_json['streams']:
            tags = rotation_json['streams'][0].get('tags', {})
            return int(tags.get('rotate', 0))
        return 0
    except Exception as e:
        print(f"Failed to get rotation: {e}")
        return 0

def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to frame based on metadata"""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def process_pose(image_rgb: np.ndarray) -> Optional[Dict]:
    """Process pose detection on RGB image"""
    results = pose_detector.process(image_rgb)
    if results.pose_landmarks:
        return {
            "landmarks": [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in results.pose_landmarks.landmark
            ]
        }
    return None

def process_hands(image_rgb: np.ndarray) -> Optional[Dict]:
    """Process hand detection on RGB image"""
    results = hands_detector.process(image_rgb)
    if results.multi_hand_landmarks:
        hands_data = []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            hands_data.append({
                "label": hand_label,
                "landmarks": [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    }
                    for lm in hand_landmarks.landmark
                ]
            })
        return {"hands": hands_data}
    return None

def process_face_mesh(image_rgb: np.ndarray) -> Optional[Dict]:
    """Process face mesh detection on RGB image"""
    results = face_mesh_detector.process(image_rgb)
    if results.multi_face_landmarks:
        faces_data = []
        for face_landmarks in results.multi_face_landmarks:
            faces_data.append({
                "landmarks": [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    }
                    for lm in face_landmarks.landmark
                ]
            })
        return {"faces": faces_data}
    return None

def process_face_detection(image_rgb: np.ndarray) -> Optional[Dict]:
    """Process face detection on RGB image"""
    results = face_detection_detector.process(image_rgb)
    if results.detections:
        faces_data = []
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            faces_data.append({
                "confidence": detection.score[0],
                "bounding_box": {
                    "x": bbox.xmin,
                    "y": bbox.ymin,
                    "width": bbox.width,
                    "height": bbox.height
                }
            })
        return {"faces": faces_data}
    return None

@app.get("/")
async def root():
    return {"message": "MediaPipe Analysis API", "endpoints": ["/pose", "/hands", "/face", "/face_mesh", "/analyze", "/pose_video", "/hands_video", "/face_video", "/analyze_video"]}

@app.post("/pose")
async def pose_from_image(file: UploadFile = File(...)):
    """Analyze pose from image"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = process_pose(img_rgb)
        
        if result is None:
            return {"pose": None, "message": "No pose detected"}
        
        return {"pose": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/hands")
async def hands_from_image(file: UploadFile = File(...)):
    """Analyze hands from image"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = process_hands(img_rgb)
        
        if result is None:
            return {"hands": None, "message": "No hands detected"}
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/face")
async def face_from_image(file: UploadFile = File(...)):
    """Analyze face detection from image"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = process_face_detection(img_rgb)
        
        if result is None:
            return {"faces": None, "message": "No faces detected"}
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/face_mesh")
async def face_mesh_from_image(file: UploadFile = File(...)):
    """Analyze face mesh from image"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = process_face_mesh(img_rgb)
        
        if result is None:
            return {"face_mesh": None, "message": "No face mesh detected"}
        
        return {"face_mesh": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze")
async def analyze_all_from_image(file: UploadFile = File(...)):
    """Analyze pose, hands, and face from image"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".heic", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process all detections
        pose_result = process_pose(img_rgb)
        hands_result = process_hands(img_rgb)
        face_result = process_face_detection(img_rgb)
        
        return {
            "pose": pose_result,
            "hands": hands_result,
            "face": face_result,
            "image_shape": {"height": img.shape[0], "width": img.shape[1]}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/pose_video")
async def pose_from_video(file: UploadFile = File(...)):
    """Analyze pose from video with iPhone rotation handling"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".m4v"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        # Get rotation from metadata
        rotation = get_video_rotation(temp_path)
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        results = []
        frame_number = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Apply rotation based on metadata
            frame = apply_rotation(frame, rotation)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = process_pose(frame_rgb)
            
            if pose_result:
                results.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps if fps > 0 else 0,
                    "pose": pose_result
                })
            
            frame_number += 1

        cap.release()
        
        return {
            "frames_processed": frame_number,
            "frames_with_pose": len(results),
            "video_info": {
                "fps": fps,
                "duration": duration,
                "rotation_applied": rotation
            },
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/hands_video")
async def hands_from_video(file: UploadFile = File(...)):
    """Analyze hands from video with iPhone rotation handling"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".m4v"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        rotation = get_video_rotation(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        results = []
        frame_number = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = apply_rotation(frame, rotation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_result = process_hands(frame_rgb)
            
            if hands_result:
                results.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps if fps > 0 else 0,
                    "hands": hands_result
                })
            
            frame_number += 1

        cap.release()
        
        return {
            "frames_processed": frame_number,
            "frames_with_hands": len(results),
            "video_info": {
                "fps": fps,
                "rotation_applied": rotation
            },
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/face_video")
async def face_from_video(file: UploadFile = File(...)):
    """Analyze faces from video with iPhone rotation handling"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".m4v"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        rotation = get_video_rotation(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        results = []
        frame_number = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = apply_rotation(frame, rotation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = process_face_detection(frame_rgb)
            
            if face_result:
                results.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps if fps > 0 else 0,
                    "faces": face_result
                })
            
            frame_number += 1

        cap.release()
        
        return {
            "frames_processed": frame_number,
            "frames_with_faces": len(results),
            "video_info": {
                "fps": fps,
                "rotation_applied": rotation
            },
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/analyze_video")
async def analyze_all_from_video(file: UploadFile = File(...)):
    """Analyze pose, hands, and faces from video with iPhone rotation handling"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".m4v"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        rotation = get_video_rotation(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        results = []
        frame_number = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = apply_rotation(frame, rotation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process all detections
            pose_result = process_pose(frame_rgb)
            hands_result = process_hands(frame_rgb)
            face_result = process_face_detection(frame_rgb)
            
            # Only include frames where at least one detection was found
            if pose_result or hands_result or face_result:
                results.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps if fps > 0 else 0,
                    "pose": pose_result,
                    "hands": hands_result,
                    "faces": face_result
                })
            
            frame_number += 1

        cap.release()
        
        return {
            "frames_processed": frame_number,
            "frames_with_detections": len(results),
            "video_info": {
                "fps": fps,
                "rotation_applied": rotation
            },
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)