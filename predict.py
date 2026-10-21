"""Step 3: predict from a single image.   Usage:  python predict.py path\\to\\photo.jpg"""
import sys, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T
import config as C

CONFIDENCE_THRESHOLD = 0.50


def load_model(path=C.MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, len(ckpt["classes"])))
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, ckpt["classes"], ckpt["img_size"]


def predict(image_path, top_k=3):
    model, classes, size = load_model()
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    conf, idx = probs.topk(top_k)
    return [(classes[i], float(c)) for c, i in zip(conf, idx)]


if __name__ == "__main__":
    results = predict(sys.argv[1])
    for name, p in results:
        print(f"{name:25s} {p*100:5.1f}%")
    if results[0][1] < CONFIDENCE_THRESHOLD:
        print("\nLow confidence - result is unreliable. Retake the photo (good light, in focus, lesion centred).")
    print("\nThis is a screening aid, not a medical diagnosis. Please consult a dermatologist.")
