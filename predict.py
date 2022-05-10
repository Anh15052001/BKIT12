
import os


import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import create_dir

""" Global parameters """
H = 512
W = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    #create_dir("test_images/mask")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    """ Load the dataset """
    data_x = glob("test_images/image/*")
    data_root = "test_images/image"

    name = "test_images/image/wp5815325"
    print(name)


    image = cv2.imread("343.png")
    h, w, _ = image.shape
    cv2.imshow("na", image)
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y = model.predict(x)[0]
    y1 = cv2.resize(y, (w, h))
    y = np.expand_dims(y1, axis=-1)
    print(y)

    """ Save the image """
    masked_image = image * y
    line = np.ones((h, 10, 3)) * 128
    print(line)
    cat_images = np.concatenate([image, line, masked_image], axis=1)

    cv2.imwrite("pre_for_gesture1.png", y1)
    cv2.imshow("test", cat_images)
    cv2.imshow("2", masked_image)

    cv2.waitKey(0)

