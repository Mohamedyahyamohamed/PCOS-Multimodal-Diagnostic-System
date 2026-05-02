# PCOS Multimodal Diagnostic System with Explainable AI (XAI) 

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://mohamedyahya72-pcos-diagnostic-engine.hf.space)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An advanced diagnostic engine for **Polycystic Ovary Syndrome (PCOS)** that integrates **Clinical Tabular Data** and **Ultrasound Imaging** using a **Decision-Level Fusion** architecture[span_0](start_span)[span_0](end_span). The system provides transparent, interpretable results through **Explainable AI (XAI)** techniques[span_1](start_span)[span_1](end_span).

---

##  Key Features
*   **Multimodal Fusion:** Combines predictions from diverse data modalities (tabular + images) to improve diagnostic reliability[span_2](start_span)[span_2](end_span).
*   **Clinical Intelligence:** Ensemble modeling using **XGBoost, LightGBM, and CatBoost** for precise analysis of patient symptoms and hormone levels[span_3](start_span)[span_3](end_span).
*   **Medical Vision:** Fine-tuned **EfficientNet-B3** for high-accuracy ultrasound scan classification (99.8% Recall)[span_4](start_span)[span_4](end_span).
*   **Explainable AI (XAI):** 
    *   **SHAP:** Provides local and global feature importance for clinical insights[span_5](start_span)[span_5](end_span).
    *   **Grad-CAM:** Generates visual heatmaps to justify ultrasound classification decisions[span_6](start_span)[span_6](end_span).
*   **Production-Ready API:** Built with **FastAPI** and containerized using **Docker** for scalable deployment[span_7](start_span)[span_7](end_span).

---

##  Tech Stack
*   **Deep Learning:** PyTorch, EfficientNet-B3, Grad-CAM[span_8](start_span)[span_8](end_span).
*   **Machine Learning:** Scikit-learn, XGBoost, CatBoost, LightGBM[span_9](start_span)[span_9](end_span).
*   **API Framework:** FastAPI, Uvicorn[span_10](start_span)[span_10](end_span).
*   **Data Processing:** Pandas, NumPy, OpenCV[span_11](start_span)[span_11](end_span).
*   **Deployment:** Docker, Hugging Face Spaces[span_12](start_span)[span_12](end_span).

---

##  System Architecture
The engine processes inputs through two parallel pipelines:
1.  **Tabular Pipeline:** Preprocesses 40+ clinical features and handles missing values dynamically using `NaN` imputation to maintain model integrity[span_13](start_span)[span_13](end_span).
2.  **Vision Pipeline:** Applies CLAHE contrast enhancement on ultrasound images before classification[span_14](start_span)[span_14](end_span).

**Decision Fusion:** The final diagnosis is determined by a weighted average of probabilities from both models, providing a comprehensive risk score[span_15](start_span)[span_15](end_span).

---

## API Documentation
The API is deployed as a microservice. You can access the interactive documentation at:
 [https://mohamedyahya72-pcos-diagnostic-engine.hf.space/docs](https://mohamedyahya72-pcos-diagnostic-engine.hf.space/docs)

### Endpoint: `/api/v1/diagnose`
*   **Method:** `POST`
*   **Payload:** `multipart/form-data`
*   **Fields:**
    *   `clinical_data` (JSON String): Optional patient clinical features[span_16](start_span)[span_16](end_span).
    *   `ultrasound_image` (File): Optional ultrasound scan image[span_17](start_span)[span_17](end_span).

---

##  Visualizations
### Grad-CAM Heatmap
The system outputs a Base64-encoded image highlighting the specific regions in the ultrasound scan that influenced the AI's diagnosis[span_18](start_span)[span_18](end_span).

### SHAP Explanations
Identifies key risk-increasing and protective factors based on clinical indicators like AMH, LH, and Follicle count[span_19](start_span)[span_19](end_span).

---
## ‍ Developed By
**Mohamed Yahya**  
*AI Engineer & Data Scientist*  
[LinkedIn](https://www.linkedin.com/in/mohamedyahya72/) | [GitHub](https://github.com/mohamedyahya72)