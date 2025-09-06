
# Face Mask Detection using CNN and TFLite


**Demo Video:** [Watch on YouTube](https://youtu.be/TIjEDFxF-e4)

**Google Colab:** [Open in Colab](https://colab.research.google.com/drive/136sglL2jlGkPiog7Bha84dDogrdxKNTy?usp=sharing)
## Dataset
The model is trained using the **Face Mask Dataset** from Kaggle:  
[Face Mask Dataset (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

It contains labeled images of people **with masks** and **without masks**, which are used for training and evaluation.
## Overview
This project provides an end-to-end solution for detecting face masks in images using a Convolutional Neural Network (CNN). The trained model is quantized and deployed using TensorFlow Lite, with an interactive web application built using Streamlit.

## Tech Stack
- Python 3.10
- TensorFlow & TensorFlow Lite
- Streamlit
- OpenCV
- NumPy
- Pillow
- Matplotlib
- scikit-learn

## Features
- Train a CNN for face mask detection
- Model quantization and deployment with TensorFlow Lite
- User-friendly web app for image-based mask detection
- Google Colab notebook for training and experimentation

## How to Run the Streamlit App
1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. Run the app:
	```bash
	streamlit run app.py
	```
3. Upload a face image to check for mask presence.

## Google Colab
For model training and exploration, use the Colab notebook (link also at the top).
## Model
The quantized TensorFlow Lite model (`face_mask_model_quantized.tflite`) is used for fast and efficient inference in the web app.

---
