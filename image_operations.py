from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def resample_img(file_loc, dim):
    img = Image.open(file_loc)
    img = img.resize(dim)
    img = np.array(img)
    return img

def resample_img_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resample_img_red(img):
    return img[:,:,0]

def resample_img_green(img):
    return img[:,:,1]

def resample_img_blue(img):
    return img[:,:,2]

def display_images(x_set, subplots, n_images, figsize=(15,2), title='', cmap=None):
    fig, axes = plt.subplots(1, subplots, figsize=figsize)
    fig.suptitle(title, fontsize=15)
    axes = axes.flatten()
    for img, ax in zip(x_set[:n_images], axes[:n_images]):
        ax.imshow(img,cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    plt.show()