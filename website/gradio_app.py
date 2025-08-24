import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# Load model
MODEL_PATH = Path('fer_vgg.h5')
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((48, 48))
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1, H, W, C)
    return x

def predict(img: Image.Image):
    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    sorted_probs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
    probs_dict = {cls: float(p) for cls, p in sorted_probs}
    top_label = CLASSES[int(np.argmax(probs))]
    return top_label, probs_dict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face image"),
    outputs=[
        gr.Label(label="Predicted Emotion"),   # will show the top label
        gr.JSON(label="Class Probabilities")   # full dict of probabilities
    ],
    title="Facial Emotion Recognition (FER2013)",
    description="Upload a face image to get the predicted emotion and per-class probabilities.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
