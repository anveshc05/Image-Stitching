#!/usr/bin/env python
# coding: utf-8

# ### Name : Anvesh Chaturvedi
# ### Roll Number : 20161094
# ### Computer Vision Assignment 2  
# ###   

import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import choice
from scipy import ndimage
import scipy
from PIL import Image
import random


THRESH_RANSAC = 0.85
NUM_ITERS = 1000


def calculateHomography(correspondences):
    ''' Computes the homography matrix based on the correspondance points given as input'''
    
    num = correspondences.shape[0]
    M = np.zeros((2*num, 9))
    for i in range(num):
        corr = correspondences[i]
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])
        M[2*i,0:3] = p1
        M[2*i,6:9] = - p2.item(0) * p1
        M[2*i + 1, 3:6] = p1
        M[2*i + 1, 6:9] = - p2.item(1) * p1

    U, D, V = np.linalg.svd(M)

    h = np.reshape(V[8], (3, 3))
    h = h / h[2,2]
    return h


def get_num_inliers(correspondances, h,  threshold):
    ''' Returns the number of inliers for currently computed homography'''
    cnt=0
    for corr in correspondances:
        point_1 = np.matrix([corr[0].item(0), corr[0].item(1), 1]).T
        point_2 = np.matrix([corr[0].item(2), corr[0].item(3), 1]).T
        p2_estimate = np.dot(h, point_1)
        p2_estimate = p2_estimate / p2_estimate.item(2)
        if np.linalg.norm(point_2 - p2_estimate) < threshold:
            cnt+=1
    return cnt
        


def ransac(corr, thresh, thresh_dist=5):
    ''' Performs RANSAC to compute a homography matrix. '''
    maxInliers = -1
    finalH = None
    num = len(corr)
    print("num: ", num)
    for i in range(NUM_ITERS):
        rand_indx = random.sample(range(num), 4)
        for j in range(4):
            if j == 0:
                random_points = corr[rand_indx[j]]
            else:
                random_points = np.vstack((random_points, corr[rand_indx[j]]))

        h = calculateHomography(random_points)

        num_inliers = get_num_inliers(corr, h, thresh_dist)
        
        if num_inliers > maxInliers:
            maxInliers = num_inliers
            finalH = h

        if maxInliers > (num*thresh):
            break
    return finalH


def crop_image(image, threshold=0):
    ''' Crops the black boundaries around the stitched images. '''
    
    flatImage = np.max(image, 2)

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def stitch(im1_rgb, im2_rgb):
    ''' Main function for stitching 2 images.
        Gets sift features for the images and constructs a list of correspondance points.
        Generates the homography matrix using RANSAC according to the correspondance points.
        Stitches the first image with warped second image.
        Crops the black boundary around stitched image.
    '''
    if len(im1_rgb.shape) != 3:
        im1_rgb = cv2.cvtColor(im1_rgb,cv2.COLOR_GRAY2RGB)
    im1 = cv2.cvtColor(im1_rgb, cv2.COLOR_RGB2GRAY)
    im1 = cv2.copyMakeBorder(im1,200,200,500,500, cv2.BORDER_CONSTANT)
    im1_rgb = cv2.copyMakeBorder(im1_rgb,200,200,500,500, cv2.BORDER_CONSTANT)
    
    if len(im2_rgb.shape) != 3:
        im2_rgb = cv2.cvtColor(im2_rgb,cv2.COLOR_GRAY2RGB)
    im2 = cv2.cvtColor(im2_rgb, cv2.COLOR_RGB2GRAY)    
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(des1, des2)

    correspondenceList = []

    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])    
    
    corrs = np.matrix(correspondenceList)
    out_ransac = ransac(corrs, THRESH_RANSAC)
    print("Homography Matrix : ")
    print(out_ransac)
    final_image = None
    
    output = np.zeros_like(im1_rgb)
    out = cv2.warpPerspective(im2_rgb, scipy.linalg.inv(out_ransac), (im1.shape[1],  im1.shape[0]))

    (x, y) = im1.shape
    for i in range(x):
        for j in range(y):
            if im1[i][j]==0 and np.sum(out[i][j])==0:
                output[i][j]=[0,0,0]
            elif im1[i][j]==0:
                output[i][j] = out[i][j]
            else:
                output[i][j] = (im1_rgb[i][j])
    
    final_image = np.copy(output)
    final_image =  np.uint8(final_image)
    final_image = crop_image(final_image)
    return final_image


def stitch_images(paths_list):
    '''Gets a list of images and returns a stitched output.'''
    num = len(paths_list)
    for i in range(num):
        im_input = np.array(Image.open(paths_list[i]))
        if i is 0:
            im = im_input
            continue
        im = stitch(im, im_input)
    return im


def display_images(path_list, output):
    ''' Displays output as desired. '''
    num, i = len(path_list), 0
    while(num > 0):
        if num == 1:
            plt.figure(figsize=(8,8))
            img = Image.open(path_list[i])
            plt.title("Image " + str(i+1))
            plt.imshow(img)
            i+=1; plt.xticks([]); plt.yticks([]);
        else:
            plt.figure(figsize=(16,16))
            plt.subplot(1,2,1)
            plt.title("Image " + str(i+1))
            plt.imshow(Image.open(path_list[i]))
            i+=1; plt.xticks([]); plt.yticks([]);
            plt.subplot(1,2,2)
            plt.title("Image " + str(i+1))
            plt.imshow(Image.open(path_list[i]))
            i+=1; plt.xticks([]); plt.yticks([]);
        num-=2
    plt.figure(figsize = (16,16))
    plt.title("Final Output")
    plt.imshow(output)
    plt.xticks([]); plt.yticks([]);
    plt.show()
            
            
# ## Results On Test Images

# Image 1


path_list = ['./test_images/img2_1.png', './test_images/img2_2.png', './test_images/img2_3.png', 
          './test_images/img2_4.png', './test_images/img2_5.png', './test_images/img2_6.png'] 
output = stitch_images(path_list)

display_images(path_list, output)


# Image 2


path_list = ['./test_images/img1_1.png', './test_images/img1_2.png'] 
output = stitch_images(path_list)

display_images(path_list, output)


# Image 3

path_list = ['./test_images/img3_1.png', './test_images/img3_2.png'] 
output = stitch_images(path_list)

display_images(path_list, output)


# Image 4

path_list = ['./test_images/img4_1.jpg', './test_images/img4_2.jpg', './test_images/img4_3.jpg'] 
output = stitch_images(path_list)

display_images(path_list, output)


# ## Results On My Images

# Image 1


path_list = ['./test_images/my_images/1_11.jpg', './test_images/my_images/1_12.jpeg'] 
output = stitch_images(path_list)

display_images(path_list, output)


# Image 2

path_list = ['./test_images/my_images/2_1.jpeg', './test_images/my_images/2_2.jpeg'] 
output = stitch_images(path_list)

display_images(path_list, output)

