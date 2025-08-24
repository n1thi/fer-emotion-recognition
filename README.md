# Facial Emotion Recognition  

This project detects human emotions from facial images using deep learning (FER2013 dataset).  
It includes a Jupyter notebook for training, a Flask web app for running predictions locally, and a live Gradio demo hosted on Hugging Face Spaces.  

---

## Contents
- `website/` – Flask web app with trained model (`fer_vgg.h5`)  
- `fer_emotion_recognition.ipynb` – Training and evaluation notebook  
- `FER_report.pdf` – Project report  
- `requirements.txt` – Dependencies  

---

## Live Demo
Try it directly in your browser with **Hugging Face Spaces**:  

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg)](https://huggingface.co/spaces/n1thi/fer-emotion-recognition)

---

## Setup & Run
1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/n1thi/fer-emotion-recognition.git
   cd fer-emotion-recognition
   pip install -r requirements.txt
    ```
2. Start the Flask app:
    ```bash
    cd website
    python app.py
    ```

## Results 

- **Dataset:** FER2013 (48×48 faces, 7 classes: angry, disgust, fear, happy, sad, surprise, neutral)

- **Best model:** VGG-13 (transfer learning + augmentation)  
  - **Top-1 Test Accuracy:** ~**62.8%**

- **Other models tried:** ResNet variant(s)  
  - **Best Validation Accuracy:** ~**47.5%**

- **Notes:**
  - Standard augmentations (flip/shift/zoom) and regularization improved generalization.  
  - Smaller/rare classes (e.g., **disgust**) are harder due to class imbalance—typical for FER2013.  
  - Optimizer state is not needed at inference; only the trained weights (`fer_vgg.h5`) are used by the Flask app.  