import subprocess
import json
import tempfile
import cv2
import numpy as np
from fastapi import UploadFile

def extract_first_frame(video_file: UploadFile) -> np.ndarray:
    # Save the uploaded file to a temporary location
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(video_file.file.read())
    temp.flush()
    temp.close()

    # Use ffprobe to read rotation metadata
    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "json", temp.name
    ]
    try:
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rotation_json = json.loads(result.stdout)
        rotation = int(rotation_json['streams'][0]['tags']['rotate']) if 'streams' in rotation_json and 'tags' in rotation_json['streams'][0] else 0
    except Exception as e:
        print("⚠️ Rotation check failed:", e)
        rotation = 0

    # Read first frame
    cap = cv2.VideoCapture(temp.name)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        return None

    # Apply rotation fix
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)