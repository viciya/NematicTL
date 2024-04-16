import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import pickle 
from natsort import natsorted
from scipy.ndimage import rotate, gaussian_filter
from scipy.stats import circmean, circstd, sem
from skimage.measure import regionprops
from scipy.ndimage import label

sys.path.append('Utilities/defect_functions') 
from defect_pairs import * 
from average_flows import *



# ----------------- Small utilities ----------------- #

def px2mic(x):
    return np.array(x) * 150 / 900

def frame2sec(x):
    return np.array(x) / 50




# ------------------- Functions --------------------- #


def compute_optical_flow(input_folder,
                         output_folder,
                         save_figures = True):


    # Scan the input folder and sort the images.
    image_list = glob.glob(r + input_folder + "\*.tif")
    image_list = natsorted(image_list, key=lambda y: y.lower())

    # Read to images and compute the optical flow.
    im_num = 100
    img1 = cv2.imread(image_list[im_num])[:,:,0]
    img2 = cv2.imread(image_list[im_num + 1])[:,:,0]
    flow = cv2.calcOpticalFlowFarneback(img1,img2,
                                        None, 0.5, 3, 
                                        winsize = 15,
                                        iterations = 3,
                                        poly_n = 5,
                                        poly_sigma = 1.2,
                                        flags = 0) 

    # Create the corresponding plots to visualize the optical flows.
    # We first improve the contrast of the images via CLAHE and then
    # overlay the optical flows on top.
    step = 15
    fig, ax1 = plt.subplots(1, 1,
                            figsize=(img1.shape[1]//100,
                                     img1.shape[0]//100))
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax1.axis('off')    
    # ax1.imshow(img1, cmap="gray")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img1)
    ax1.imshow(img_clahe, cmap="gray")  

    if save_figures:
        fig.savefig(save_folder + "/raw.png",
                    bbox_inches=extent.expanded(1.15, 1.15))

        field = np.stack((flow[:,:,0], flow[:,:,1]), axis=-1)
        flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=15)
        flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=15)        
        vorticity = gaussian_filter(curl_npgrad(field), sigma=15)
        # c = ax1.imshow(vorticity, cmap='RdBu', alpha=.3)
        # fig.colorbar(c, ax=ax1)

        vort_min = 0.005
        vortTFr = vorticity > vort_min
        vortTFl = vorticity < vort_min
        ur = flow[:,:,0] * vortTFr
        vr = flow[:,:,1] * vortTFr
        ul = flow[:,:,0] * vortTFl
        vl = flow[:,:,1] * vortTFl

        y, x = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
        # ax1.quiver(x[::step, ::step], y[::step, ::step], 
        #         flow[::step, ::step, 0], -flow[::step, ::step, 1], 
        #         color="red", scale=120, alpha=.9, width=.003)

        ax1.quiver(x[::step, ::step], y[::step, ::step], 
                   ur[::step, ::step], -vr[::step, ::step], 
                   color="b", scale=120, alpha=.8, width=.002, minlength=0)
        ax1.quiver(x[::step, ::step], y[::step, ::step], 
                   ul[::step, ::step], -vl[::step, ::step], 
                   color="r", scale=120, alpha=.8, width=.002, minlength=0)


        # speed = np.sqrt(flow[:,:,0]**2 + flow[:,:,0]**2)
        # lw = 3*speed / speed.max()
        # ax1.streamplot(x[::step, ::step],y[::step, ::step],
        #         flow[::step, ::step, 0], flow[::step, ::step, 1], 
        #         density=4., color='white', linewidth=lw[::step, ::step])

        # if save_figures:
        #     fig.savefig(save_folder + "/flow_field.png",
        # bbox_inches=extent.expanded(1.15, 1.15))
