
from random import randrange
import numpy as np
import os
import cv2
from torch import randint
import uuid

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.02
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
def blur_img(img, factor = 70):
   kW = int(img.shape[1] / factor)
   kH = int(img.shape[0] / factor)
    
   #ensure the shape of the kernel is odd
   if kW % 2 == 0: kW = kW - 1
   if kH % 2 == 0: kH = kH - 1
    
   blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
   return blurred_img

path = ".\\data\\train"
for filename in os.listdir(os.path.join(path, "image")):
    img_path = os.path.join(path, "image",filename)
    mask_path = os.path.join(path, "mask", filename)
    img = cv2.imread(img_path)
    msk = cv2.imread(mask_path)
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    # break
    rand_n = randrange(2)
    if rand_n == 1:
        cv2.imwrite(img_path,blur_img(img))
    else:
        uuid_new = str(uuid.uuid4())
        print(img_path.rsplit('.', 1)[0])
        new_img_filename = f"{img_path.rsplit('.', 1)[0]}_{uuid_new}.png"
        new_mask_filename = f"{mask_path.rsplit('.', 1)[0]}_{uuid_new}.png"
        # print(new_img_filename)
        
        # break
        cv2.imwrite(new_img_filename,noisy("s&p",img))
        cv2.imwrite(new_mask_filename,msk)

