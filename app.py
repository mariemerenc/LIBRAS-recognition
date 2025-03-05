import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

st.title("Reconhecimento de LIBRAS")


@st.cache_resource
def load_custom_model():
    return load_model('libras_model_v2.keras')

model = load_custom_model()


@st.cache_data
def load_labels():
    return np.load('labels.npy', allow_pickle=True)

labels = load_labels()


def predict_image(img):
    img = img.resize((50, 50))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    preds = model.predict(img_array)
    pred_label = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100 
    
    return pred_label, confidence


uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption='Imagem carregada', use_column_width=True)

    pred_label, confidence = predict_image(img)

    st.success(f"Predição: **{pred_label}**")
    st.info(f"Confiança: **{confidence:.2f}%**")