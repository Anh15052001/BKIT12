
from train import load_data
from metrics import dice_loss, dice_coef, iou
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import pandas as pd
import cv2
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

""" Global parameters """
H = 256
W = 256
PATH_ROOT = '.data/image'
PATH_RES = '.data/res'


""" Creating a directory """


# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     return sorted(blob(path))
# def save_results(image, mask, y_pred, save_image_path):


def save_results(y_pred, save_image_path): 
    y_pred = np.array(y_pred, dtype='uint8')
    y_pred = cv2.resize(y_pred, (256,256))
    y_pred = y_pred * 255
    y_pred = cv2.resize(y_pred, (256,256))
    cv2.imwrite(save_image_path, y_pred)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    """ Load the dataset """
    # dataset_path = "new_data"
    # valid_path = os.path.join(dataset_path, "test")
    # test_x, test_y = load_data(valid_path)
    # print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    list_dir = os.listdir(PATH_ROOT)
    for filename in list_dir:
        print(f"read {filename} {list_dir.index(filename)}/{len(list_dir)}")
        """ Reading the image """
        image = cv2.imread(os.path.join(PATH_ROOT, filename))
        h, w, _ = image.shape
        image = cv2.resize(image, (W, H))
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        # mask = cv2.imread("sample2.png", cv2.IMREAD_GRAYSCALE)
        # mask = cv2.resize(mask, (W, H))

        """ Prediction """
        y_pred = model.predict(x)[0]

        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_results(y_pred, os.path.join(PATH_RES, filename))
   
