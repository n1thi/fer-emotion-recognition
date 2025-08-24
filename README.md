# Facial Emotion Recognition for Enhanced Communication

A deep learning project that identifies facial emotions and returns accessible feedback, aimed at assisting visually impaired users. The repo includes training notebooks, a saved model, and a minimal Flask API for inference.

> Key points: FER2013 dataset, transfer learning (VGG-13/ResNet), augmentation, and a Flask-based prediction endpoint. Final test accuracy for VGG-13: **~62.8%**; ResNet val accuracy: **~47.5%**.

## Contents

- `notebooks/` — your original exploration and training notebooks (add your `.ipynb` here)
- `report/` — project report PDF
- `models/` — exported/saved models (`.h5` or SavedModel format)
- `app/` — minimal Flask app for inference
- `requirements.txt` — Python dependencies
- `.gitignore` — common ignores for Python/conda/notebooks
- `LICENSE` — *(optional; add your preferred license, e.g., MIT)*

## Quickstart (Local)

1) **Create and activate a virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) **Install dependencies**:
```bash
pip install -r requirements.txt
```

3) **Put your model file** in `models/` as `vgg_emotion.h5` (or adjust the path in the app).

4) **Run the Flask API**:
```bash
export FLASK_APP=app/main.py  # Windows PowerShell: $env:FLASK_APP="app/main.py"
flask run --port 7860
```
Then POST an image to `http://127.0.0.1:7860/predict` or open the simple web form at `/`.

### Test the API
```bash
curl -X POST http://127.0.0.1:7860/predict   -F "file=@/path/to/face.jpg"
```

**Response**
```json
{"filename":"face.jpg","emotion":"happy","probs":{"angry":0.01,"disgust":0.00,"fear":0.05,"happy":0.88,"sad":0.02,"surprise":0.03,"neutral":0.01}}
```

## Minimal Flask App (in `app/main.py`)

Below is a reference implementation that expects a 48×48 (or any size) face image and runs the loaded Keras model.
You can adapt this to your preprocessing. Save as `app/main.py`:

```python
import io, os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf

# ---- Load model once ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/vgg_emotion.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def preprocess(img: Image.Image) -> np.ndarray:
    # Convert to RGB, resize to model's expected input, normalize [0,1]
    img = img.convert("RGB").resize((48,48))
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1,H,W,C)
    return x

app = Flask(__name__)

INDEX_HTML = '''
<!doctype html>
<title>FER Demo</title>
<h2>Facial Emotion Recognition</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value="Predict">
</form>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error":"empty filename"}), 400

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    return jsonify({
        "filename": file.filename,
        "emotion": CLASSES[top],
        "probs": {cls: float(p) for cls, p in zip(CLASSES, probs)}
    })
```

> If your model expects grayscale or different normalization, adjust `preprocess()` accordingly.

## Training (Notebook)

1) Place the FER2013 data in a folder structure compatible with your generators (e.g., `data/train/<class>/...`, `data/val/<class>/...`).  
2) Open your notebook under `notebooks/` and run training cells.  
3) Save the best model to `models/vgg_emotion.h5` and reuse it with the Flask app.

## Results (from the report)

- **VGG-13**: Test accuracy ~**62.77%**, test loss ~**1.3455**.  
- **ResNet**: Best validation accuracy ~**47.54%**, validation loss ~**1.3584**.  

These align with a classic FER2013 baseline with transfer learning and augmentation.

## Deploying a Live Demo

You have three solid options:

### 1) Hugging Face Spaces (Gradio/Streamlit) — easiest
- Wrap your model with a small Gradio UI instead of Flask (or keep both).
- Add `app_gradio.py` and `requirements.txt`, then create a new Space (select Gradio).
- Push your repo; Spaces will build and host the demo for free (with usage limits).

### 2) Render / Railway / Fly.io — keeps Flask
- Add a `Procfile` like: `web: gunicorn app.main:app`
- Create a free web service, set `MODEL_PATH` and deploy from GitHub.

### 3) Docker anywhere (AWS EC2, GCP, Azure)
- Add a `Dockerfile` (see below), build & run on any VM/container platform.

**Example `Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY models ./models
ENV MODEL_PATH=models/vgg_emotion.h5
ENV PORT=7860
EXPOSE 7860
CMD ["python","-m","flask","--app","app/main.py","run","--host","0.0.0.0","--port","7860"]
```

## Repo Structure Template

```
.
├── app/
│   └── main.py
├── models/
│   └── vgg_emotion.h5           # <— your trained weights
├── notebooks/
│   └── prabhasv_nithired_himavenk.ipynb
├── report/
│   └── prabhasv_nithired_himavenk_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Acknowledgments

- FER2013 dataset.
- Keras/TensorFlow models (VGG, ResNet) with augmentation and early stopping.