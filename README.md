# PCOS Multimodal Diagnostic System with Explainable AI (XAI)

This system provides an integrated diagnostic engine for Polycystic Ovary Syndrome (PCOS) by combining clinical patient data with ultrasound imaging. It utilizes a decision-level fusion architecture to deliver precise risk assessments and interpretable medical justifications.

## Project Links
*   Live Application (Hugging Face): https://huggingface.co/spaces/mohamedyahya72/PCOS-Diagnostic-Engine
*   API Documentation (Swagger UI): https://mohamedyahya72-pcos-diagnostic-engine.hf.space/docs

## Key Features
*   **Multimodal Fusion**: Combines predictions from diverse data modalities (tabular and images) to improve diagnostic reliability.
*   **Clinical Intelligence**: Employs ensemble modeling using XGBoost, LightGBM, and CatBoost for precise analysis of patient symptoms and hormone levels.
*   **Medical Vision**: Utilizes a fine-tuned EfficientNet-B3 model for high-accuracy ultrasound scan classification, achieving 99.8% Recall.
*   **Explainable AI (XAI)**:
    *   **SHAP**: Provides local and global feature importance to gain clinical insights from tabular data.
    *   **Grad-CAM**: Generates visual heatmaps to justify ultrasound classification decisions by highlighting relevant pathological regions.
*   **Production-Ready API**: Built with FastAPI and containerized using Docker for scalable and efficient deployment.

## Technical Stack
*   **Backend Framework**: FastAPI and Uvicorn.
*   **Machine Learning**: Scikit-learn, XGBoost, CatBoost, and LightGBM.
*   **Deep Learning**: PyTorch and EfficientNet-B3.
*   **Data Processing**: OpenCV, Pandas, and NumPy.
*   **Infrastructure**: Docker and Hugging Face Spaces.

## API Specification
The diagnostic engine is exposed via a primary endpoint:
*   **Endpoint**: `POST /api/v1/diagnose`
*   **Payload Type**: `multipart/form-data`
*   **Parameters**:
    *   `clinical_data` (String): A JSON formatted string containing clinical features.
    *   `ultrasound_image` (File): An ultrasound scan image file.

## Author
**Mohamed Yahya**  
AI Engineer & Data Scientist  
[[LinkedIn Profil](https://www.linkedin.com/in/mohamedyahyamohamed/)] | [[GitHub Profil](https://github.com/Mohamedyahyamohamed)]
