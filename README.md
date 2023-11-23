# Hand Gesture Classification

## Introduction

A real-time system for hand gesture classification. In this project, I use `Mediapipe` to inference and LSTM model from `Tensorflow` to train model.

## Setting up

Firstly, we need to install all required libraries

``` bash
pip install -r requirements.txt
```

## Use case

This project has 3 primary parts

### Generate data

This part is used for generating data to train model. Run the `generate_data.py` to start. The generated data will be store in `data` folder

### Train

After generating data, run the `train.py` to retrain the model. In this task, all classes in `data` folder will be labels for classification model. The model will be stored in `models/model.h5`

### Main

Run `main.py` for starting the application. The model in `models/model.h5` will be loaded to inference the hand gesture.