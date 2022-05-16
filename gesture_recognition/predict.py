
import cv2
import mediapipe as mp
#Chúng ta phải cho ra 2 thread
#thread chính là hiển thị ảnh
#tuy nhiên nếu cứ 10 timestep liên tiếp hiển thị sẽ bị lỗi đa luồng
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import os
model=tf.keras.models.load_model('model/checkpoint_BKIT12.hdf5')
#khởi tạo camera


#khởi tạo mediapipe
mpPose=mp.solutions.pose
pose=mpPose.Pose()
#Khởi tạo vẽ pose
mpDraw=mp.solutions.drawing_utils
lm_list=[]
def make_pose(results):
    #print(results.pose_landmarks.landmark)
    c_lm=[]
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img
class_name={0:"hand_fand", 1:'look_at_me', 2:'peekaboo', 3:'reverse_signal', 4:'scissor', 5:'scratch', 6:'typing', 7:'up_down', 8:'VAR', 9:'wave_hand'}
def detect(model, lm_list):
    #chuyển lm_list về tensor
    global label
    lm_list=np.array(lm_list)
    lm_list=np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results=model.predict(lm_list)
    #print('heloo', results)
    label=class_name[np.argmax(results)]
    print(label)

#hàm vẽ label lên
i=1
label='...'
def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
time_step=13
root='public_test_data/public_test_gesture_data/data'
file_name = []
predict = []
for file in os.listdir(root):

    print(file)
    file_name.append(file)
    cap = cv2.VideoCapture(root+"/"+file)
    while True:
        ret, frame=cap.read()
        if ret:
            #chuyển ảnh về RGB
            frameRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results=pose.process(frameRGB)
            i+=1
            if i > 1:
                if results.pose_landmarks:
                    lm=make_pose(results)
                    lm_list.append(lm)
                    if len(lm_list)==time_step:
                        t=threading.Thread(target=detect, args=(model, lm_list))
                        t.start()
                        lm_list=[]
                    #vẽ các điểm khung xương lên
                    frame=draw_landmark_on_image(mpDraw, results, frame)
            frame=draw_class_on_image(label, frame)
            cv2.imshow('image', frame)

            if cv2.waitKey(1)==ord('q'):
                break
        else:
            print("no frame")
            break
    predict.append(label)
id = []
num_id = len(predict)
for i in range(num_id):
    id.append(1)
results = pd.DataFrame({'id': id, 'file_name': file_name, 'label': predict})
results.to_csv('results.csv', index=False)

cap.release()
cv2.destroyAllWindows()