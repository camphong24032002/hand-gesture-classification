import numpy as np
import pandas as pd
from utils import *

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

def preprocess_data(df, classes):
    num_classes = len(classes)
    X = []
    y = []
    
    for idx, class_name in enumerate(classes):
        class_df = df[class_name]
        num_sample = class_df.shape[0]
        one_hot_encoding = [0] * num_classes
        one_hot_encoding[idx] = 1
        for i in range(num_sample-DATA_RANGE):
            X.append(class_df.iloc[i: i+DATA_RANGE])
            y.append(one_hot_encoding)

    return (np.array(X), np.array(y))

def get_model(num_classes, input_shape):
    model = Sequential([
        LSTM(units = 50, return_sequences = True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),
        LSTM(units = 50),
        Dropout(0.2),
        Dense(units = num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer = 'adam',
        metrics = ['accuracy'],
        loss = 'categorical_crossentropy'
    )
    return model

def main():
    df = load_data()
    classes = load_classes()
    num_classes = len(classes)
    X, y = preprocess_data(df, classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = get_model(num_classes, input_shape)
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    model.save('../models/model.h5')

if __name__ == "__main__":
    main()