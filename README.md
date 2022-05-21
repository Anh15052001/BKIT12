# BKIT12
 BK_Naver competition
1. Segmentation
- The first you can read README.md file in each folder and following it to prepare run
- After you done, follow this  tutorial
    1.1 Run file training to train: 
        python train.py
    After that, you will have file checkpoint model.h5 and you can use it to predict or eval data.
    
    1.2 To predict, you need model pretrained : model.h5 in folder files or you can download here: https://drive.google.com/file/d/1BFVYCdWympC9TvhjklyxuMQRkWPOzL2_/view?usp=sharing
        python predict.py
    After that, you will have data preprocess for gesture. 

    1.3 To get result about task1 : segmentation body, you can run file predict with model pretrained and data test
        python eval.py
    
    After that, result will in folder contains mask predict. To get file results.csv for task 1 you need create file results.csv and run
        python get_result_segment.py
2. Gesture recogntion
- 2.1 The first, you must install media pipe
MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. MediaPipe Python package is available on PyPI for Linux, macOS and Windows.

You can, for instance, activate a Python virtual environment:

$ python3 -m venv mp_env && source mp_env/bin/activate
Install MediaPipe Python package and start Python interpreter:

(mp_env)$ pip install mediapipe
(mp_env)$ python3
In Python interpreter, import the package and start using one of the solutions:

import mediapipe as mp
Mpose = mp.solutions.pose
- 2.2 The second, you will create skeleton point, then save to file text
- Run file gesture_recognition/make_data.py
- You can download video data from gesture_data: https://drive.google.com/drive/folders/1YmGRa7QkOU6syYzLX3M20mu9oK-rsEeW
 and download video preprocess from body segmentation: https://drive.google.com/drive/folders/1wWfQoURpJe7hWWgYYl2_ttMj88iafzwH?fbclid=IwAR3ssDj6NrNt8U-2GMdTqwz_kQ6QVUA_LtEA26Aj6PSuy0IfyVXkaIJ1uM0
- You will save skeleton parameters into the gesture_recognition/data_skeleton 
- You can download data_skeleton from here: https://drive.google.com/drive/folders/1xVv3tnDGA8gbbvBXvijLdUV5L5duKMq-?usp=sharing
- 2.3 The third, you run file gesture_recognition/train_gesture.py with path data
from data_skeleton at step 2.2 so that divide timesteps
- You can download model from here: https://drive.google.com/file/d/1-1A4RQ2fRlHLCQJOJ6-FQnb7Xx6-3tQj/view?usp=sharing
- 2.4: The fourth, you run file gesture_recogntion/predict.py so that have results 
- NOTE: you must run least 200 epochs

