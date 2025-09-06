import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Face Mask Detection (TFLite)", layout="centered")
st.title("Face Mask Detection (TFLite)")
st.write("Upload a face image to check for mask presence.")

@st.cache_resource
def load_tflite_interpreter():
    interpreter = tf.lite.Interpreter(model_path="face_mask_model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_interpreter()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img_array = np.array(image.resize((128, 128)), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    pred_label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if pred_label == 0:
        st.success(f"Mask detected (Confidence: {confidence:.2%})")
    else:
        st.error(f"No mask detected (Confidence: {confidence:.2%})")

st.markdown("---")
st.markdown("Face Mask Detection powered by TensorFlow Lite and Streamlit.")
