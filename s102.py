import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# 모델 파일 경로
model_path = 'my_model.h5'

# 모델 훈련 및 저장
if not os.path.exists(model_path):
    from model import train_model
    train_model()  # 모델이 없으면 훈련

# 모델 로드
model = load_model(model_path)

# Streamlit 앱 제목
st.title("딥러닝 모델 예측기")

# 사용자 입력
input_data = []
for i in range(10):
    value = st.number_input(f"특성 {i + 1}의 값을 입력하세요:", value=0.0)
    input_data.append(value)

# 예측 버튼
if st.button("예측하기"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    st.write(f"예측 결과: {prediction[0][0]:.4f}")
