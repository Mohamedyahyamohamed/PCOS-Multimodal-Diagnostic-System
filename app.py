import os
import json
import re
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

# ==========================================
# 1. System Initialization
# ==========================================
app = FastAPI(title="PCOS Multimodal Diagnostic System", version="1.0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Model Loading (Clinical & Ultrasound)
# ==========================================
# تحميل نموذج التحاليل السريرية
try:
    clinical_model = joblib.load('PCOS_FINAL_CHAMPION.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load clinical model: {e}")

FEATURES = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 'RR (breaths/min)',
    'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)', 'No. of aborptions',
    'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)',
    'Waist:Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)',
    'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)',
    'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)'
]

def clean_col_names(df):
    new_cols = []
    for col in df.columns:
        cleaned = re.sub(r'\s+', '_', col.strip())
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned)
        cleaned = re.sub(r'_{2,}', '_', cleaned).strip('_')
        new_cols.append(cleaned)
    df.columns = new_cols
    return df

# تحميل نموذج الأشعة
try:
    weights = models.EfficientNet_B3_Weights.DEFAULT
    vision_model = models.efficientnet_b3(weights=weights)
    num_ftrs = vision_model.classifier[1].in_features
    vision_model.classifier[1] = nn.Linear(num_ftrs, 2)
    vision_model.load_state_dict(torch.load('pcos_model_b3_advanced.pth', map_location=device))
    vision_model.to(device)
    vision_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load vision model: {e}")

vision_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Processing Functions
# ==========================================
def process_clinical_data(data_dict: dict) -> float:
    df = pd.DataFrame([data_dict])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    df = df[FEATURES]
    df_cleaned = clean_col_names(df.copy())
    
    return float(clinical_model.predict_proba(df_cleaned)[0][1])

async def process_ultrasound_image(file: UploadFile) -> float:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Invalid image format.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image)

    input_tensor = vision_transform(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = vision_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        return probabilities[1].item()

# ==========================================
# 4. API Endpoints
# ==========================================
@app.get("/")
def health_check():
    return {"status": "Active", "message": "PCOS Diagnostic Engine is running."}

@app.post("/api/v1/diagnose")
async def diagnose_patient(
    clinical_data: str = Form(..., description="JSON string of patient clinical features"),
    ultrasound_image: UploadFile = File(..., description="Patient ultrasound scan")
):
    try:
        try:
            clinical_dict = json.loads(clinical_data)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON format in clinical_data."})

        # استخراج النتائج من النموذجين
        clinical_risk = process_clinical_data(clinical_dict)
        ultrasound_risk = await process_ultrasound_image(ultrasound_image)

        # دمج القرارات (Average Fusion)
        fusion_risk = (clinical_risk + ultrasound_risk) / 2.0
        
        # تصنيف الحالة
        if fusion_risk >= 0.7:
            diagnosis = "High Risk (PCOS Detected)"
        elif fusion_risk >= 0.4:
            diagnosis = "Moderate Risk (Needs Clinical Review)"
        else:
            diagnosis = "Low Risk (Healthy)"

        return {
            "status": "Success",
            "diagnosis": diagnosis,
            "overall_risk_score": round(fusion_risk, 4),
            "breakdown": {
                "clinical_indicators_risk": round(clinical_risk, 4),
                "ultrasound_analysis_risk": round(ultrasound_risk, 4)
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Processing Error", "details": str(e)})