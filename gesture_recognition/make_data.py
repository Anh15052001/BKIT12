"""
Người dùng thực hiện hành động -> đọc ảnh từ webcam -> sử dụng thư việ pose estimate ->
lưu trữ các điểm skleton -> đưa vào csv -> tạo model train->
"""
import cv2
import pandas as pd
import mediapipe as mp
#Khởi tạo camera
import os
data = "Video"



#khởi tạo thư viện mediapine
#khởi tạo thư viện lưu trữ các điểm (x, y, z, visibility)
Mpose=mp.solutions.pose
pose=Mpose.Pose()
#Khởi tạo thư viện để vẽ các điểm pose lên
myDraw=mp.solutions.drawing_utils
#list lưu trữ thông số các điểm
lm_list=[]
#số frame dùng để đọc

label='wave_hand'
#hàm ghi nhận các thông số khung xương
def make_pose(results):
    print(results.pose_landmarks.landmark)
    c_lm=[]
    for id, c in enumerate(results.pose_landmarks.landmark):
        c_lm.append(c.x)
        c_lm.append(c.y)
        c_lm.append(c.z)
        c_lm.append(c.visibility)
    return c_lm

#hàm để vẽ các điểm skeleton
def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, Mpose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img
for file in os.listdir(data):
    print("Xử lí "+data+'/'+file)

    cap = cv2.VideoCapture(data+'/'+file)
    while (cap.isOpened()):
        #đọc ảnh
        ret, frame=cap.read()
      #  frame = cv2.resize(frame, (700, 500))
        if ret:
            #cv2 ảnh là BGR phải chuyển sang thư viện pose estimate xử lí là RGB

            frameRGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #xử lí qua thư viện pose estimate để lấy các điểm ghi nhận thông số khung xương
            results=pose.process(frameRGB)

            if results.pose_landmarks:
                lm=make_pose(results)
                lm_list.append(lm)
                #vẽ khung xưong lên để xem
                frame=draw_landmark_on_image(myDraw, results, frame)
        else:
            print("No frame")
            break
        #hiển thị ảnh
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #Giai phóng cam

    cap.release()
    cv2.destroyAllWindows()

#chuyển về dataframe
df=pd.DataFrame(lm_list)
df.to_csv("data_skeleton/"+label+'.txt')

