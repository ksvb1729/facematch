import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from deepface import DeepFace
import insightface
from insightface.app import FaceAnalysis
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load InsightFace model
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

@app.get("/")
async def root():
    return {"message": "Face Analyzer API is running"}

def extract_face_embedding(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    faces = face_app.get(img)
    if faces:
        return faces[0].embedding, img, faces[0]
    return None, img, None

def estimate_fatigue_level(emotion):
    fatigue_emotions = ["sad", "neutral", "tired", "fear"]
    return "High" if emotion.lower() in fatigue_emotions else "Low"

def analyze_skin_tone(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_skin_tone = hsv_img[:, :, 0].mean()  # hue average
    if avg_skin_tone < 10:
        return "Very pale"
    elif avg_skin_tone < 20:
        return "Pale"
    elif avg_skin_tone < 30:
        return "Normal"
    elif avg_skin_tone < 40:
        return "Warm"
    else:
        return "Dark"

def detect_smile(emotion):
    return emotion.lower() == "happy"

def detect_attention_level(emotion):
    return "Low" if emotion.lower() in ["tired", "sad", "fear"] else "High"

def detect_health_warnings(age, emotion, fatigue_level):
    warnings = []
    if fatigue_level == "High":
        warnings.append("Possible fatigue or sleep issues")
    if emotion.lower() in ["sad", "fear"]:
        warnings.append("Signs of stress or low mood")
    if age > 50:
        warnings.append("Consider regular health check-ups")
    return warnings

def estimate_bmi_from_face(face):
    if not face:
        return "Unknown"
    bbox = face.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    ratio = width / height
    if ratio > 0.85:
        return "High"
    elif ratio > 0.70:
        return "Moderate"
    else:
        return "Low"

def facial_symmetry_score(face):
    if not face or len(face.kps) < 5:
        return "Unknown"
    left_eye, right_eye = face.kps[0], face.kps[1]
    nose = face.kps[2]
    mouth_left, mouth_right = face.kps[3], face.kps[4]
    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_distance = np.linalg.norm(mouth_left - mouth_right)
    nose_mid = (left_eye + right_eye) / 2
    symmetry_error = abs(nose[0] - nose_mid[0]) / eye_distance
    return round(1.0 - symmetry_error, 2)

# Helper function to ensure float32 is converted to Python float
def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    return value

@app.post("/compare-faces/")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    try:
        logger.debug("Received request to compare faces.")
        
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()

        # Log image sizes
        logger.debug(f"Image 1 size: {len(img1_bytes)} bytes")
        logger.debug(f"Image 2 size: {len(img2_bytes)} bytes")
        
        emb1, img1, face1 = extract_face_embedding(img1_bytes)
        emb2, img2, face2 = extract_face_embedding(img2_bytes)

        # Log face detection results
        if face1:
            logger.debug("Face 1 detected.")
        else:
            logger.debug("No face detected in Image 1.")
        
        if face2:
            logger.debug("Face 2 detected.")
        else:
            logger.debug("No face detected in Image 2.")

        if emb1 is None or emb2 is None:
            return JSONResponse(content={"error": "Face not detected in one or both images."}, status_code=400)

        from numpy.linalg import norm
        dot = np.dot(emb1, emb2)
        cos_sim = dot / (norm(emb1) * norm(emb2))
        match_score = float((cos_sim + 1.0) / 2.0)
        is_match = match_score >= threshold

        analysis1 = DeepFace.analyze(img1, actions=["age", "gender", "emotion"], enforce_detection=False)
        gender1 = "Man" if analysis1[0]["gender"]["Man"] > analysis1[0]["gender"]["Woman"] else "Woman"
        emotion1 = analysis1[0]["dominant_emotion"]
        age1 = int(analysis1[0]["age"])
        fatigue1 = estimate_fatigue_level(emotion1)
        skin_tone1 = analyze_skin_tone(img1)
        smile1 = detect_smile(emotion1)
        attention1 = detect_attention_level(emotion1)
        health_warnings1 = detect_health_warnings(age1, emotion1, fatigue1)
        bmi1 = estimate_bmi_from_face(face1)
        symmetry1 = facial_symmetry_score(face1)

        analysis2 = DeepFace.analyze(img2, actions=["age", "gender", "emotion"], enforce_detection=False)
        gender2 = "Man" if analysis2[0]["gender"]["Man"] > analysis2[0]["gender"]["Woman"] else "Woman"
        emotion2 = analysis2[0]["dominant_emotion"]
        age2 = int(analysis2[0]["age"])
        fatigue2 = estimate_fatigue_level(emotion2)
        skin_tone2 = analyze_skin_tone(img2)
        smile2 = detect_smile(emotion2)
        attention2 = detect_attention_level(emotion2)
        health_warnings2 = detect_health_warnings(age2, emotion2, fatigue2)
        bmi2 = estimate_bmi_from_face(face2)
        symmetry2 = facial_symmetry_score(face2)

        # Convert numpy float32 to standard float for serialization
        result = {
            "image1": {
                "age": age1,
                "gender": gender1,
                "dominant_emotion": emotion1,
                "fatigue_level": fatigue1,
                "skin_tone": skin_tone1,
                "smiling": smile1,
                "attention_level": attention1,
                "bmi_level": bmi1,
                "facial_symmetry_score": convert_to_float(symmetry1),
                "health_warnings": health_warnings1
            },
            "image2": {
                "age": age2,
                "gender": gender2,
                "dominant_emotion": emotion2,
                "fatigue_level": fatigue2,
                "skin_tone": skin_tone2,
                "smiling": smile2,
                "attention_level": attention2,
                "bmi_level": bmi2,
                "facial_symmetry_score": convert_to_float(symmetry2),
                "health_warnings": health_warnings2
            },
            "match_score": round(match_score * 100, 2),
            "face_match": bool(is_match)
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
