
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm

from metrics import dice_coef, dice_loss, iou

""" Global parameters """
H = 256
W = 256
RES_ROOT_PATH = './predata_for_gesture'
# VIDEO_ROOT_PATH = './gesture_training_data/gesture_data'
IMAGE_ROOT_PATH = './membrane/data'
def sub_background(path, file, filename, image):
    h, w, _ = image.shape
    # cv2.imshow("na", image)
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y = model.predict(x)[0]
    y1 = cv2.resize(y, (w, h))
    y = np.expand_dims(y1, axis=-1)
    # print('3')

    """ Save the image """
    masked_image = image * y
    cv2.imwrite(os.path.join(path, file, filename), masked_image)

def prepare_data_image_root(path):
    for file in os.listdir(path):
        if not file.endswith('mask'):
            print(f"file: {file}")
            for filename in os.listdir(os.path.join(path, file)):

                print(f"filename: {filename} ")
                if not os.path.exists(os.path.join(RES_ROOT_PATH, file)):
                    os.makedirs(os.path.join(RES_ROOT_PATH, file))
                image = cv2.imread(os.path.join(path, file, filename))
                sub_background(RES_ROOT_PATH, file, filename, image)

# not handle now
def prepare_data_video_root(path):
    for file in os.listdir(path):
        for vid in os.listdir(os.path.join(path, file)):
            cap = cv2.VideoCapture(os.path.join(path, file, vid))
            cnt = 0
            while(cap.isOpened()):
                print(f"{vid}_cnt")

                ret, frame = cap.read()
                if ret:
                    if not os.path.exists(os.path.join(RES_ROOT_PATH, 'video', file, vid)):
                        os.makedirs(os.path.join(RES_ROOT_PATH, 'video', file, vid))
                    sub_background(os.path.join(RES_ROOT_PATH, 'video'), os.path.join(file, vid), f"{vid}_{cnt}.png", frame )
                cnt+=1
            break

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    #create_dir("test_images/mask")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    # prepare_data_video_root(VIDEO_ROOT_PATH)                
    prepare_data_image_root(IMAGE_ROOT_PATH)                


