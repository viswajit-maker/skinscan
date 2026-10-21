# SkinScan – AI Skin Condition Screening Website

An integrated end-to-end web application that connects the PyTorch deep learning model (`outputs/best_model.pt`) with a responsive web interface (`index.html`) using a lightweight Flask backend server (`app.py`).

---

## 🚀 Quick Start

### 1. Launch with One Click
Double-click the **`start_website.bat`** file in the project folder.

### 2. Or Launch via Command Line
Activate the virtual environment and start `app.py`:

```powershell
.\venv\Scripts\activate
python app.py
```

### 3. Open in Browser
Visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎯 Features

1. **Integrated Architecture**:
   - Single server serves both the frontend website (`/`) and the AI prediction API (`/predict`).
   - PyTorch model weights are loaded once in memory upon server startup for fast inference.
   - Cross-Origin Resource Sharing (CORS) enabled.

2. **Frontend Capabilities**:
   - **Live Camera Feed**: Capture photos directly from a webcam or phone camera.
   - **Upload Photo**: Drag-and-drop or file picker for image files.
   - **Instant AI Analysis**: Real-time confidence breakdown across all 6 target classes.
   - **Clinical Descriptions & Guidance**: Practical educational info and care tips for identified conditions.
   - **Confidence Warnings & Medical Disclaimer**: Guides user when image quality is low or confidence is uncertain.

3. **Recognized Conditions (6 Classes)**:
   - Atopic Dermatitis (AD)
   - Contact Dermatitis
   - Eczema
   - Scabies
   - Seborrheic Dermatitis
   - Tinea Corporis (Ringworm)

---

## 🛠️ API Endpoints

- `GET /` — Serves the `index.html` web application.
- `GET /health` — Returns JSON with server status, model readiness, and recognized disease classes.
- `POST /predict` — Accepts multipart form-data with key `file` (image). Returns sorted class probabilities and clinical guidance.
