import glob
import os
import random
from random import shuffle

import imageio
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal, ndimage as ndimage
import cv2
# from TrafficL_Manager import train_imgs, train_gt, val_imgs, val_gt, test_imgs, test_gt
from skimage import exposure, feature
from skimage.restoration import denoise_nl_means, estimate_sigma


def readImage(path):
    return imageio.imread(path)


def getNeighbors(image_shape, indices, windown = 80):
    cluster_map = np.zeros(image_shape[:2])
    clusters = []
    for indice in indices:
        if cluster_map[indice[0], indice[1]]:
            continue
        else:
            cluster_map[indice[0] - windown//2: indice[0] + windown//2, indice[1] - windown//2: indice[1] + windown//2] = 1
            clusters.append([indice])

    return cluster_map, np.asarray(clusters).reshape((len(clusters),2))


gt_dir = r'C:\Users\RENT\Desktop\mobileye\CityScapes\gtFine'
imgs_dir = r'C:\Users\RENT\Desktop\mobileye\CityScapes\leftImg8bit'
gt_train_path = os.path.join(gt_dir, 'train')
imgs_train_path = os.path.join(imgs_dir, 'train')
gt_val_path = os.path.join(gt_dir, 'val')
imgs_val_path = os.path.join(imgs_dir, 'val')
gt_test_path = os.path.join(gt_dir, 'test')
imgs_test_path = os.path.join(imgs_dir, 'test')




def get_files(imgs_dir, gt_dir):
    cities = os.listdir(imgs_dir)
    gt = []
    imgs = []
    for city in cities:
        new_gt_path = os.path.join(gt_dir, city)
        new_imgs_path = os.path.join(imgs_dir, city)
        gt += glob.glob(os.path.join(new_gt_path, "*labelIds.png"))
        imgs += glob.glob(os.path.join(new_imgs_path, "*.png"))
    imgs.sort()
    gt.sort()
    return imgs, gt

train_imgs, train_gt = get_files(imgs_train_path, gt_train_path)
val_imgs, val_gt = get_files(imgs_val_path, gt_val_path)
test_imgs, test_gt = get_files(imgs_test_path, gt_test_path)

def prepare_ground_truth(img):
    new_image = np.zeros((img.shape[0], img.shape[1], 3))
    new_image[img == 19, :] = [0, 255, 0]
    return new_image.astype(np.float32)


def data_generator(batch_size=1, num_classes=1, mode='train', imgs=train_imgs, gt=train_gt, im_size=500):
    # Expects sorted lists of training images and ground truth as
    # 'data' and 'labels'
    if (mode == 'val'):
        imgs = val_imgs
        gt = val_gt
    elif (mode == 'test'):
        imgs = test_imgs
        gt = test_gt

    # get shape from any image
    shape_im = imageio.imread(random.choice(imgs))

    # Shuffle dataset
    combined = list(zip(imgs, gt))
    shuffle(combined)
    imgs[:], gt[:] = zip(*combined)

    while (True):
        for i in range(0, len(imgs), batch_size):
            images = np.empty((batch_size, im_size, im_size, shape_im.shape[2]))
            labels = np.empty((batch_size, im_size, im_size, 3))
            for j, img in enumerate(imgs[i:i + batch_size]):
                # Crop the size we want from a random spot in the image (as a form of
                # minor data augmentation)
                new_start_row = np.random.randint(0, shape_im.shape[0] - im_size)
                new_start_col = np.random.randint(0, shape_im.shape[1] - im_size)
                train_im = imageio.imread(img).astype(np.float32)

                train_im = train_im[new_start_row:new_start_row + im_size, new_start_col:new_start_col + im_size]
                images[j, :, :, :] = train_im

                gt_im = imageio.imread(gt[i + j])
                gt_im = gt_im[new_start_row:new_start_row + im_size, new_start_col:new_start_col + im_size]
                labels[j, :, :, :] = prepare_ground_truth(gt_im)

            yield (images, labels)


def visualize_prediction(ims, gts):
    fig = plt.figure(figsize=(20, 20))
    for i, im in enumerate(ims):
        a = fig.add_subplot(5, 2, i + 1)
        new_image = Image.blend(Image.fromarray(im.astype(np.uint8), mode='RGB').convert('RGBA'),
                                Image.fromarray(gts[i].astype(np.uint8), mode='RGB').convert('RGBA'),
                                alpha=0.5)
        plt.imshow(new_image)
    plt.show()


def AnalyzeImageClassic(img, show = True):
    fig = plt.figure(figsize=(100, 100))
    general_clusters = []
    for i in range(3):
        a = fig.add_subplot(4, 3, i + 1)
        red = np.zeros(img.shape, dtype='uint8')
        red[:, :, i] = img[:, :, i]
        red_edge = ProcessChannel(img[:, :, i])
        general_clusters.append(red_edge)
        plt.scatter(red_edge[:,1], red_edge[:,0])
        plt.imshow(red)

    general_clusters = np.array(general_clusters)
    general_clusters = np.vstack(general_clusters)
    print("Total: ", len(general_clusters))
    cluster_map, clusters = getNeighbors(red.shape, general_clusters)

    a = fig.add_subplot(4, 3, 4)
    plt.imshow(img/255)
    plt.scatter(clusters[:,1], clusters[:,0])

    fig.add_subplot(4,3,5)
    plt.imshow(cluster_map)
    print("After Hedging: ", len(clusters))
    # sums = np.zeros(cluster_map.shape)
    # sums[np.where(cluster_map)] = img[np.where(cluster_map)].argmax(axis = -1)
    # sums[np.where(cluster_map)] = np.histogram(sums.flatten()).
    if(show):
        plt.show()
    return clusters


def ProcessChannel(image, threshold_val = 0.1):
    scharr = np.array([[-1/9, -1/9, -1/9],
                       [-1/9, 8/9, -1/9],
                       [-1/9, -1/9, -1/9]])
    # Adaptive Equalization
    # img_adapteq =
    # logspace = np.log(image + 1)
    # sigma_est = np.mean(estimate_sigma(image, multichannel=False))

    patch_kw = dict(patch_size=3,  # 5x5 patches
                    patch_distance=5    ,  # 13x13 search area
                    multichannel=False)
    logspace = image
    # logspace = denoise_nl_means(image, h=0.6 * sigma_est, fast_mode=True,
    #                                 **patch_kw)
    grad = (signal.convolve2d(logspace, scharr, boundary='symm', mode='same'))**2
    # grad = cv2.Canny(image,400 , 600)
    # plt.figure()
    # plt.imshow(grad)
    normalized =  grad/ np.max(grad)
    threshold = normalized
    result = ndimage.maximum_filter(threshold, size=10)
    threshold = (np.logical_and(threshold == result, threshold >= threshold_val))
    threshold = np.logical_and(threshold,  feature.canny(image, sigma=3))
    indices = np.argwhere(threshold)
    return indices


# dat_gen = data_generator(batch_size=10, im_size=1000)

# In[8]:


# ims, gts = dat_gen.__next__()


# In[10]:
# path = r"C:\Users\RENT\Desktop\mobileye\CityScapes\leftImg8bit\test\berlin\berlin_000000_000019_leftImg8bit.png"
path = r"C:\Users\RENT\Downloads\astrophysics\tubingen_000002_000019_leftImg8bit.png"
# path = r"C:\Users\RENT\Downloads\astrophysics\frankfurt_000001_007973_leftImg8bit.png"
g_im = imageio.imread(path)
AnalyzeImageClassic(g_im)
# visualize_mag(g_im) 1373 377
# edges2 = feature.canny(g_im[:,:,1], sigma=3)