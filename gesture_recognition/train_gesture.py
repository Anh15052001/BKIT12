import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import os
X=[]
y=[]
timestep=13
# đọc dữ liệu
data_handfand = "data_skeleton/hand_fand"
for file in os.listdir(data_handfand):
    print("Xử lí: ", file)
    dataset_hand_fand = pd.read_csv(data_handfand + "/" + file)
    dataset_1 = dataset_hand_fand.iloc[:, 1:].values  # chuyển về matran
    print(dataset_1)
    # chia timestep

    num_s1 = len(dataset_1)
    print("Num of value: ", num_s1)
    for i in range(timestep, num_s1):
        X.append(dataset_1[i - timestep:i, :])
        y.append(0)
#đọc dữ liệu
data_lookatme = "data_skeleton/look_at_me"
for file in os.listdir(data_lookatme):
  print("Xử lí: ", file)
  dataset_look_at_me=pd.read_csv(data_lookatme+"/"+file)
  dataset_2=dataset_look_at_me.iloc[:, 1:].values #chuyển về matran
  print(dataset_2)
  #chia timestep


  num_s2=len(dataset_2)
  print("Num of value: ", num_s2)
  for i in range(timestep, num_s2):
      X.append(dataset_2[i-timestep:i, :])
      y.append(1)
data_peekaboo = "data_skeleton/peekaboo"
for file in os.listdir(data_peekaboo):
  print("Xử lí: ", file)
  dataset_peekaboo=pd.read_csv(data_peekaboo+"/"+file)
  dataset_3=dataset_peekaboo.iloc[:, 1:].values
  print(dataset_3)
  num_s3=len(dataset_3)
  print("Num of value: ", num_s3)
  for i in range(timestep, num_s3):
      X.append(dataset_3[i-timestep:i, :])
      y.append(2)
data_reverse = "data_skeleton/reverse_signal"
for file in os.listdir(data_reverse):
  print("Xử lí: ", file)
  dataset_reverse=pd.read_csv(data_reverse+"/"+file)
  dataset_4=dataset_reverse.iloc[:, 1:].values
  print(dataset_4)
  num_s4=len(dataset_4)
  print("Num of value: ", num_s4)
  for i in range(timestep, num_s4):
      X.append(dataset_4[i-timestep:i, :])
      y.append(3)
data_scissor = "data_skeleton/scissor"
for file in os.listdir(data_scissor):
  print("Xử lí: ", file)
  dataset_scissor=pd.read_csv(data_scissor+"/"+file)
  dataset_5=dataset_scissor.iloc[:, 1:].values
  print(dataset_5)
  num_s5=len(dataset_5)
  print("Num of value: ", num_s5)
  for i in range(timestep, num_s5):
      X.append(dataset_5[i-timestep:i, :])
      y.append(4)
data_scratch = "data_skeleton/scratch"
for file in os.listdir(data_scratch):
  print("Xử lí: ", file)
  dataset_scratch=pd.read_csv(data_scratch+"/"+file)
  dataset_6=dataset_scratch.iloc[:, 1:].values
  print(dataset_6)
  num_s6=len(dataset_6)
  print("Num of value: ", num_s6)
  for i in range(timestep, num_s6):
      X.append(dataset_6[i-timestep:i, :])
      y.append(5)
data_typing = "data_skeleton/typing"
for file in os.listdir(data_typing):
  print("Xử lí: ", file)
  dataset_typing=pd.read_csv(data_typing+"/"+file)
  dataset_7=dataset_typing.iloc[:, 1:].values
  print(dataset_7)
  num_s7=len(dataset_7)
  print("Num of value: ", num_s7)
  for i in range(timestep, num_s7):
      X.append(dataset_7[i-timestep:i, :])
      y.append(6)
data_updown = "data_skeleton/up_down"
for file in os.listdir(data_updown):
  print("Xử lí: ", file)
  dataset_up=pd.read_csv(data_updown+"/"+file)
  dataset_8=dataset_up.iloc[:, 1:].values
  print(dataset_8)
  num_s8=len(dataset_8)
  print("Num of value: ", num_s8)
  for i in range(timestep, num_s8):
      X.append(dataset_8[i-timestep:i, :])
      y.append(7)
data_VAR = "data_skeleton/VAR"
for file in os.listdir(data_VAR):
  print("Xử lí: ", file)
  dataset_var=pd.read_csv(data_VAR+"/"+file)
  dataset_9=dataset_var.iloc[:, 1:].values
  print(dataset_9)
  num_s9=len(dataset_9)
  print("Num of value: ", num_s9)
  for i in range(timestep, num_s9):
      X.append(dataset_9[i-timestep:i, :])
      y.append(8)
data_wavehand = "data_skeleton/wave_hand"
for file in os.listdir(data_wavehand):
  print("Xử lí: ", file)
  dataset_wave=pd.read_csv(data_wavehand+"/"+file)
  dataset_10=dataset_wave.iloc[:, 1:].values
  print(dataset_10)
  num_s10=len(dataset_10)
  print("Num of value: ", num_s10)
  for i in range(timestep, num_s10):
      X.append(dataset_10[i-timestep:i, :])
      y.append(9)
X=np.array(X)
y=np.array(y)
print(X.shape)
#chia data
from sklearn.preprocessing import LabelBinarizer
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
lb=LabelBinarizer()
y_train=lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)
from keras.callbacks import ModelCheckpoint
#xây dựng model
model=Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
filepath='model/checkpoint_epoch40.hdf5'
callback=ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
model.fit(X_train, y_train, batch_size=32, epochs=150, verbose=1, validation_data=(X_test, y_test), callbacks=[callback])
model.save('model/model_epoch40.h5')
