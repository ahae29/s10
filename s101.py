import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    # 데이터 생성
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, size=(1000, 1))

    # 모델 정의
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10,)))
    model.add(Dense(1, activation='sigmoid'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 훈련
    model.fit(X, y, epochs=10, batch_size=32)

    # 모델 저장
    model.save('my_model.h5')

if __name__ == "__main__":
    train_model()
