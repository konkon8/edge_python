import cpseg_functions as cps

import numpy as np
import pandas as pd

from PIL.Image import fromarray
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.segmentation import clear_border
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import ruptures as rpt
import cv2
from cv2 import morphologyEx, MORPH_CLOSE

from joblib import Parallel, delayed
from glob import glob
import time
from random import randint

def ak_get_filename(imdata_dir):
    """
    Obtain an image file list with *_C0.tif in the specified directory.

    Parameters
    ----------
    imdata_dir : str
        A path to the directory containing image files to be processed.

    Returns
    -------
    imFilename : str
        A list of image files.

    """
    imFilename = sorted(glob(imdata_dir + '*_C0.tif'))
    return imFilename

def add_noise(img, noise_density):
    """
    add_noise(img,density)
    Created on Thu Oct  7 17:12:23 2021
    
    Salt-and-pepper noise can be applied only to greyscale images

    Modified from
    https://www.geeksforgeeks.org/add-a-salt-and-\
        pepper-noise-to-an-image-with-python/

    """
    	
    row , col = img.shape # Get the dimensions of the image
    number_of_pixels =int(( row * col ) * noise_density)
    
    # Randomly pick some pixels in the image for coloring them white (max)
    for i in range(number_of_pixels):
        y_coord=randint(0, row - 1) # Pick a random y coordinate		
        x_coord=randint(0, col - 1) # Pick a random x coordinate
        img[y_coord][x_coord] = np.max(img) # Color that pixel to max
    
    # Randomly pick some pixels in the image for coloring them black
    for i in range(number_of_pixels):
        y_coord=randint(0, row - 1) # Pick a random y coordinate
        x_coord=randint(0, col - 1) # Pick a random x coordinate
        img[y_coord][x_coord] = 0 # Color that pixel to black
    
    return img


def rot_edge(irot,values):
    # Variables
    rot_dgree_list, I, edge_mean, numBorder, model, y_interval, penalty_value, changepoint_algo = values
    I_Height, I_Width = I.shape  # image shape

    # Image rotation
    rot_degree = rot_dgree_list[irot]  # Rotation angle(degree) clockwise:minus
    im = fromarray(I)  # Convert to pillow object
    J_plt = im.rotate(rot_degree, expand=True, fillcolor=edge_mean)  # Rotate
    J_np = np.array(J_plt)  # Convert to array
    J_Height, J_Width = J_np.shape  # Size of rotated image
    J = J_np
    #print(F'Angle {irot} degree rotated'.format(irot=irot))

    # Border detection
    BW1 = cps.ak_get_edge(J, numBorder, model, y_interval, penalty_value, changepoint_algo)

    # Inverse rotation of the border
    BW1_imobject = fromarray(BW1)

    inv_rot_degree = np.copysign(rot_degree, -1).astype(np.int64)
    BW1_imobject_rot = BW1_imobject.rotate(inv_rot_degree, expand=True)  # inverse rotate
    BW = np.array(BW1_imobject_rot).astype(int)  # Convert to int nparray

    # Crop a rectangle at the image center to the size of the original image
    if rot_degree == 0 or rot_degree == 180:  # If the size of the rotated image is same as the original
        BW12 = BW1
    else:
        left = int((BW.shape[1] - I_Width) / 2)  # J_Width # Positions of the corners
        top = int((BW.shape[0] - I_Height) / 2)  # J_Height
        right = left + I_Width
        bottom = top + I_Height
        BW12 = BW[top:bottom, left:right]  # Crop a rectangle at the image center

    return BW12

def ak_rotate_edge_overlay(I,numLayer,rotation_angle,numBorder,model,y_interval,penalty_value, changepoint_algo):
    """
    Created on Fri Apr 16 07:26:17 2021

    'ak_rotate_edge_overlay' rotate an image, detect edges, and inversely
    rotate the edge. If multiple rotation angles were specified, each edges
    were added or stacked along a new axiｓ.'I' must be an 2-dimensional array
    which represent gray scale image.
    """
    #%%

    # parameters
    # numLayer       = 0  # 0: single
    #                     # 1: multiple
    # rotation_angle = 1  # 0: 45 degree, 3: semicircle
    #                     # 1: 30 degree, 4: semicircle
    #                     # 2: 15 degree, 5: semicircle
    # numBorder      = 2  # Number of change points
    # dY             = 3  # Hight of each rectangle
    # region         = 0  # 0: calculate all y, 1: calculate 1/yRatio along y
    # yRatio         = 5  # if region = 1,calculate 1/yRatio along y

    # Processing
    # Rotation angle (degree)

    if rotation_angle == 0:
        rot_degree_list = np.array([0,45,90,135,180,225,270,315]) # Every 45˚
    elif rotation_angle == 1:
        rot_degree_list = np.array([0,30,60,90,120,150,180,210,240,270,300,330]) # Every 30˚
    elif rotation_angle == 2:
        rot_degree_list = np.array([0,15,30,45,60,75,90,105,120,135,150,165,180,195,\
                           210,225,240,255,270,285,300,315,330,345]) # Every 15˚
    elif rotation_angle == 3:
        rot_degree_list = np.array([0,45,90,135]) # Every 45˚, semicircle
    elif rotation_angle == 4:
        rot_degree_list = np.array([0,30,60,90,120,150]) # Every 30˚, semicircle
    elif rotation_angle == 5:
        rot_degree_list = np.array([0,15,30,45,60,75,90,105,120,135,150,165]) # Every 15˚, semicircle

    # Mean pixel value at the edge of the image
    img_ori = I
    I = np.array(I)
    edge = np.hstack((I[9,:], I[-10,:], I[:,9].T, I[:,-10].T))
    edge_mean = int(np.mean(edge).item())

    BW_final = np.zeros(I.shape)          # empty border object

    values = rot_degree_list, I, edge_mean, numBorder, model, y_interval, penalty_value, changepoint_algo

    result = Parallel(n_jobs=-1)([delayed(rot_edge)(i_rot, values) for i_rot in np.arange(len(rot_degree_list))])
    # Overlay images
    bw_each = np.array(result)
    bw_overlay = np.sum(bw_each, axis=0)
    if numLayer == 0:
        BW_final = bw_each
    elif numLayer == 1:
        BW_final = bw_overlay
    else:
        print('The variable "numLayer" should be either 0 or 1.')

    # Normalize(binalize)
    BW_final = np.where(BW_final != 0, 1, 0)  # Replace non-zero elements to 1

    return BW_final

def ak_get_edge_xpostion(iy,values):
    # Crop image as rectangle w/ hight of one pixel
    I, numBorder, model, xBorder, penalty_value, changepoint_algo = values
    intensity_iy = I[iy, :]  # iy: y position

    # Detect change points(x)
    algo = rpt.KernelCPD(kernel=model, min_size=2, jump=1).fit(intensity_iy) # default: min_size=2
    if changepoint_algo == 1: # Number of border position not known
        my_bkps = algo.predict(pen=penalty_value)
        #
        repeat_bkps = 1
        my_bkps = algo.predict(pen=penalty_value)
        while repeat_bkps in np.arange(30) and len(my_bkps) - 1 > my_bkps[-1] /10:
            print(f"repeat_bkp:{repeat_bkps}")
            print(f"len(my_bkps):{len(my_bkps)}")
            my_bkps = algo.predict(pen=penalty_value)
            repeat_bkps += 1

        xBorder = np.floor(np.array(my_bkps[0:-1])) # Avoid total indexes at the end

        if xBorder.size == 0:
            xpos = 0
        else:
            xpos = xBorder[0:xBorder.size].T
    elif changepoint_algo == 0: # Number of border position fixed
        my_bkps = algo.predict(n_bkps=numBorder)
        #xBorder = np.floor(np.array(my_bkps) + 1)
        xBorder = np.floor(np.array(my_bkps[0:-1])) # Avoid total indexes at the end
        xpos = xBorder[0:numBorder].T

    return xpos

def ak_get_edge(I, numBorder, model, y_interval, penalty_value, changepoint_algo):
    """
    Created on Fri Apr 16 07:08:16 2021

    'ak_get_edge' detects edges in an image.
    """

    t_e0 = time.time()
    # Arguments
    Iori = np.array(I)  # Convert to image to numpy array
    I = Iori

    # Initialization of variables
    if changepoint_algo == 0: # border number fixed
        numBorder = numBorder
    elif changepoint_algo == 1: # border number not known
        numBorder = I.shape[1] # width of the original image
    xBorder = np.zeros(numBorder)  # edge position x coordinates
    xChangePoint = np.zeros((I.shape[0], numBorder))  # edge position coordinates [y,x]

    # Gaussian blur
    sigma = 3
    I_blur = cv2.GaussianBlur(I, (5,1), sigma)  # Gaussian blur
    I = I_blur

    # Processing
    values = [I, numBorder, model, xBorder, penalty_value, changepoint_algo]
    y_scan_index = np.arange(0, I.shape[0], y_interval)

    # Edge detection
    result = np.zeros((len(y_scan_index),numBorder)).astype(int)
    ind_list = np.arange(0,len(y_scan_index))
    for i in ind_list:
        xpos_i = np.array(ak_get_edge_xpostion(y_scan_index[i], values)).astype(int)
        result[i, 0:xpos_i.size] = xpos_i
    xChangePoint[y_scan_index, 0:numBorder] = result
    xChangePoint = xChangePoint.astype(int)

    # Get border coordinate
    subVector_h1 = np.tile(np.arange(I.shape[0]), numBorder)
    subVector_h2 = np.ravel(xChangePoint, order='F').astype(np.int64)
    BW = np.zeros((I.shape[0], I.shape[1]))
    my_index = np.arange(len(subVector_h1)).astype(np.int64)
    for i in my_index:
        BW[subVector_h1[i], subVector_h2[i]] = 1

    t_e1 = time.time()
    elapsed_time = t_e1 - t_e0
    print(f'Edge detection time(sec): {elapsed_time}')

    return BW

def ak_post_process(BW,smallNoiseRemov,senoise,neib,select_biggest,nskeletonize):
    """
    Post processing edge maps by removing small noises etc.
    
    Parameters
    ----------
    BW : numpyndarray, int
         2D Edge map
    smallNoiseRemov : int
         0:No noise removal
         1:Connect border, then remove small noises
         2:Remove small noisess first, then connect
    senoise : int
         Structural element size
    neib : int
         Area of imclose, pix
    select_biggest : int
         0: do not select
         1: select biggest
    nskeletonize : int
         0: No skeletonization
         1: Skeletonize
    
    Returns BWlast
    -------
    
    """
    # Remove signal at the edge
    BW = clear_border(BW) # remove artifacts connected to image border
   
    # Remove small objects
    BW = BW.astype('uint8')
    se_noiseremove = disk(senoise) #structual elements, circle    
    if smallNoiseRemov == 0: # No noise removal
        BWs = BW
    elif smallNoiseRemov == 1:  #Connect border -> Remove small noise      
        BW1 = morphologyEx(BW, MORPH_CLOSE, se_noiseremove) # Connect border
        BW1_labels = label(BW1)
        BWs = remove_small_objects(BW1_labels, min_size=neib) # Remove small object
       
    elif smallNoiseRemov == 2:  # Remove small noises -> Connect border
        BW_labels = label(BW)
        BW1 = remove_small_objects(BW_labels, min_size=neib) # Remove small object
        BWs = morphologyEx(BW1, MORPH_CLOSE, se_noiseremove) # Connect border
    else:
        print('smallNoiseRemov should be 0,1,2.')
    BWs = np.where(BWs != 0, 1, 0)
   
    # Select largest component (segmentation)
    if select_biggest == 0:
        BWs_biggest = BWs
    elif select_biggest == 1:
        label_BWs = label(BWs, return_num=True)  # Label connected components
        label_image = label_BWs[0]  # labeled image
        properties = ['label', 'area']
        df = pd.DataFrame(regionprops_table \
                              (label_image, properties=properties))  # Data frome of area and label
        df_area_max = np.max(df.area)  # Area of largest component
        max_index = np.array(np.where(df.area == df_area_max))  # label of the largest component
        label_image_largest = np.where(label_image == (max_index + 1), 1,
                                       0)  # replace the largest region w/ 1 and others with 0
        BWs_biggest = np.where(label_image_largest != 0, 1, 0)  # Replace non-zero elements to 1
    else:
        print('BWs_biggest should be 0 or 1.')

    # Reduce object to 1-pixel wide curved lines
    if nskeletonize == 0:
        BW_skel = BWs_biggest
    elif nskeletonize == 1:
        BW_skel = skeletonize(BWs_biggest)
    else:
        print('nskeletonize should be 0 or 1.')    
    
    BWlast = BW_skel
    
    return BWlast

def pratt(Ea,Ed):
    """
    Created on Tue Oct 12 09:58:15 2021
    
    @author: kondow_a
    
    Converted from
    Vivek Bhadouria (2021). Pratt's Figure of Merit 
    (https://www.mathworks.com/matlabcentral/fileexchange
     /60473-pratt-s-figure-of-merit), 
    MATLAB Central File Exchange. Retrieved October 12, 2021.
    
    Function EDPM : Edge detector performance measure function. 
    	          Calculates for a given edge image the false alarm 
    		  count, miss count and figure of merit (F) values.   		 	
     
        Input(s)... Ea : Actual edge image (ground truth)    	
    		 Ed : Detected edge image.
    
        Output(s).. fac: False alarm count
    		 msc: miss count
    		 F  : Figure of merit 
    """
    Ea = Ea.astype(float) # ground truth image
    Ed = Ed.astype(float) # detected edge image
    N,M = Ea.shape
    if np.any(Ea.shape != Ed.shape):
        print('Actual and detected edge image sizes must be same')
    
    a=0.1 # edge shift penalty constant
    fac_ind = np.nonzero(Ea-Ed==-1)
    fac = np.max(np.shape(fac_ind))     # False Alarm Count
    msc_ind = np.nonzero(Ea-Ed==1)
    msc = np.max(np.shape(msc_ind))     # Miss Count
    Na = np.sum(Ea) # int in matlab vs float
    Nd = np.sum(Ed) # int in matlab vs float
    if Nd == 0:
        F = 0
    else:
        c = 1/np.maximum(Na,Nd)

        ia, ja = np.nonzero(Ea==1) # indeces in python are matlab - 1
        Aes = np.zeros(Na.astype(int))
        for l in np.arange(0, Na, dtype='int'):
            Aes[l] = Ed[ia[l],ja[l]]
        #Aes = np.reshape(Aes, (1,Na.astype(int) ))# convert to row array

        mi = ia[np.nonzero(Aes==0)]
        mj = ja[np.nonzero(Aes==0)]
        F = c * np.sum(Aes)

        for k in np.arange(0,np.max(mi.shape)):
            n1 = 0
            n2 = 0
            m1 = 0
            m2 = 0
            while np.sum(Ed[(mi[k]-n1):(mi[k]+n2+1), (mj[k]-m1):(mj[k]+m2)+1]) < 1: # check!
                # print(f'k = {k}, mi[k] = {mi[k]}, mj[k] = {mj[k]}, n1 = {n1}\
                # , n2 = {n2}, m1 = {m1}, m2 = {m2}\
                # , fom sum = {np.sum(Ed[(mi[k]-n1):(mi[k]+n2+1), (mj[k]-m1):(mj[k]+m2)+1])}')
                if mi[k] - n1 > 1:
                    n1 = n1 + 1
                if mi[k] + n2 < N-1:
                    n2 = n2 + 1
                if mj[k] - m1 > 1:
                    m1 = m1 + 1
                if mj[k] + m2 < M-1:
                    m2 = m2 + 1

            di = np.max(np.array([n1,n2,m1,m2]))
            #F = F + np.round(c/(1 + (a * np.square(di))),4)
            F = F + c/(1 + (a * np.square(di)))

        F = F * 100
    
    return F, fac, msc

def ak_edge_eval(BWgt, BW):
    """
    Compare an edge map with a ground truth
    Ground truth edge map should be provided as an 2D array
    """

    # define score
    dtype = [('fom','float64'),('mse','float64'),('psnr','float64')] # 倍精度浮動小数点数
    score = np.zeros(1, dtype = dtype)
    #score = np.zeros(1)
    
    # transform imput data type
    BWgt = BWgt.astype('uint8')
    BW = BW.astype('uint8')
    
    # Evaluation
    score['fom']  = cps.pratt(BWgt,BW)[0] # Pratt's figure of merit (FOM)
    score['mse']  = mean_squared_error(BWgt,BW)
    score['psnr'] = peak_signal_noise_ratio(BWgt,BW)
    
    return score
