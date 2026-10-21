import io
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms as T
import config as C

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Load model weights on startup
print(f"Loading model weights from {C.MODEL_PATH}...")
if not C.MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {C.MODEL_PATH}. Please ensure training has generated best_model.pt.")

ckpt = torch.load(C.MODEL_PATH, map_location="cpu")
classes = ckpt["classes"]
img_size = ckpt.get("img_size", C.IMG_SIZE)

# EfficientNet-B0 backbone matching training architecture
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(classes))
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Image preprocessing matching training/prediction pipeline
transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"Model successfully loaded. Recognized classes ({len(classes)}): {classes}")

DISEASE_INFO = {
    "Atopic Dermatitis": {
        "description": "A chronic, itchy inflammatory skin condition often linked with allergies or asthma.",
        "advice": "Keep skin moisturized, avoid harsh soaps, and consult a dermatologist for anti-inflammatory treatments."
    },
    "Contact Dermatitis": {
        "description": "Skin inflammation triggered by direct contact with an irritating chemical or allergen.",
        "advice": "Identify and avoid contact with the trigger substance. Cool compresses and topical creams can soothe irritation."
    },
    "Eczema": {
        "description": "A broad category of skin conditions characterized by dry, itchy, red, or cracked patches.",
        "advice": "Use gentle emollient moisturizers, avoid known irritants or temperature extremes, and consult a doctor."
    },
    "Scabies": {
        "description": "A contagious, intensely itchy infestation caused by microscopic burrowing mites.",
        "advice": "Requires prescription scabicide treatment for the patient and close household contacts, plus laundering bedding."
    },
    "Seborrheic Dermatitis": {
        "description": "A common condition causing flaky scales and redness, mainly in oil-rich areas like scalp, face, and chest.",
        "advice": "Medicated antifungal or anti-dandruff shampoos and mild topical treatments are commonly used."
    },
    "Tinea Corporis": {
        "description": "A superficial fungal infection typically causing a ring-shaped, itchy red rash (ringworm).",
        "advice": "Keep the area dry and clean. Topical antifungal creams are commonly recommended."
    }
}

@app.route("/")
def index():
    """Serves the integrated frontend page."""
    return send_from_directory(C.PROJECT_ROOT, "index.html")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming API status and model readiness."""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "classes": classes
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns sorted disease prediction probabilities."""
    if "file" not in request.files:
        return jsonify({"error": "No 'file' field found in multipart form data."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "Empty or missing file."}), 400

    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Top predictions sorted descending
        confidences, indices = probabilities.topk(len(classes))
        predictions = []
        for conf, idx in zip(confidences, indices):
            cls_name = classes[idx]
            info = DISEASE_INFO.get(cls_name, {})
            predictions.append({
                "label": cls_name,
                "confidence": round(float(conf), 4),
                "description": info.get("description", ""),
                "advice": info.get("advice", "")
            })

        return jsonify({
            "predictions": predictions,
            "top_prediction": predictions[0]
        })

    except Exception as e:
        return jsonify({"error": f"Error analyzing image: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"SkinScan Website is live at: http://127.0.0.1:{port}")
    print(f"Open http://127.0.0.1:{port} in your browser.")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
