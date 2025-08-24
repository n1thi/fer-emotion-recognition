# Facial Emotion Recognition  

This project detects human emotions from facial images using deep learning (FER2013 dataset).  
It includes a Jupyter notebook for training and a Flask web app for running predictions.  

---

## Contents
- `website/` – Flask web app with trained model (`fer_vgg.h5`)  
- `fer_emotion_recognition.ipynb` – Training and evaluation notebook  
- `FER_report.pdf` – Project report  
- `requirements.txt` – Dependencies  

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