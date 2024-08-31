import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('my_model.h5')

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
