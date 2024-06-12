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
from matplotlib.colors import ListedColormap

sys.path.append('Utilities/defect_functions') 
from defect_pairs import * 
from average_flows import *



# ----------------- Small utilities ----------------- #

def px2mic(x):
    return np.array(x) * 150 / 900

def frame2sec(x):
    return np.array(x) / 50


def calculate_okubo_weiss(u, v):
    dudx, dudy = np.gradient(u)
    dvdx, dvdy = np.gradient(v)
    OW = (dudx - dvdy) ** 2 + (dudy + dvdx) ** 2 - (dudx + dvdy) ** 2
    return OW


def calculate_vorticity(u, v):
    dudx, dudy = np.gradient(u)
    dvdx, dvdy = np.gradient(v)
    return dvdy - dudx


def detect_vortices_area(ow, th_factor=6.):
    # Detect vortices
    vortex_mask = ow > th_factor*np.mean(ow)
    labeled_image,_ = label(vortex_mask)
    regions = regionprops(labeled_image)
    return [prop.area for prop in regions]


def detect_vortices_area_and_vorticity(u,v, ow, th_factor=6.):
    vorticity = calculate_vorticity(u, v)
    vorticity = vorticity / np.mean(vorticity**2)
    vortex_mask = ow > th_factor*np.median(ow_field)
    labeled_image,_ = label(vortex_mask)

    regions = regionprops(labeled_image, intensity_image=vorticity)

    return [prop.area for prop in regions], \
        [region.mean_intensity for region in regions]


def imagelist_to_votrex_area(img_list, sigma=35):
    """same as def imagepair_to_votrex_area()
        but input is image list
        average flow dlield over several images
    """
    flow = time_average(img_list[:-1], img_list[1:])

    u = gaussian_filter(flow[...,0], sigma=sigma)
    v = gaussian_filter(flow[...,1], sigma=sigma)

    # Calculate the Okubo-Weiss parameter
    ow_field = calculate_okubo_weiss(u, v)

    # return detect_vortices_area(ow_field)
    return detect_vortices_area_and_vorticity(u,v, ow_field)


def imagepair_to_votrex_area(img1, img2, sigma=35):

    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0) 
    u = gaussian_filter(flow[...,0], sigma=sigma)
    v = gaussian_filter(flow[...,1], sigma=sigma)

    # Calculate the Okubo-Weiss parameter
    ow_field = calculate_okubo_weiss(u, v)

    # return detect_vortices_area(ow_field)
    return detect_vortices_area_and_vorticity(u,v, ow_field)


def time_average(img_list1, img_list2):
    flows = []
    for im1, im2 in zip(img_list1, img_list2):
        img1 = cv2.imread(im1)[:,900:,0]
        img2 = cv2.imread(im2)[:,900:,0]
        flows.append(
            cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0) 
        )
    u = np.concatenate(flows, axis=-1)[:,:,::2].mean(axis=-1)
    v = np.concatenate(flows, axis=-1)[:,:,1::2].mean(axis=-1)
    
    return np.stack((u, v), axis=-1)


# ------------------- Functions --------------------- #


def compute_optical_flow(input_folder,
                         output_folder,
                         save_figures = True):


    # Scan the input folder and sort the images.
    image_list = glob.glob(input_folder + "/*.tif")
    image_list = natsorted(image_list, key=lambda y: y.lower())
        
    # Read to images and compute the optical flow.
    im_num = 0
    img1 = cv2.imread(image_list[im_num])[:,:,0]
    img2 = cv2.imread(image_list[im_num + 1])[:,:,0]
    flow = cv2.calcOpticalFlowFarneback(img1, img2,
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

        # fig.savefig(output_folder + "/raw.png",
        #             bbox_inches=extent.expanded(1.15, 1.15))
        
        # speed = np.sqrt(flow[:,:,0]**2 + flow[:,:,0]**2)
        # lw = 3*speed / speed.max()
        # ax1.streamplot(x[::step, ::step],y[::step, ::step],
        #         flow[::step, ::step, 0], flow[::step, ::step, 1], 
        #         density=4., color='white', linewidth=lw[::step, ::step])

        # if save_figures:
        #     fig.savefig(save_folder + "/flow_field.png",
        # bbox_inches=extent.expanded(1.15, 1.15))

        if save_figures:
            fig.savefig(output_folder + "/optical_flow.png",
                        bbox_inches=extent.expanded(1.15, 1.15))

    # Clean up.
    del img1
    del img2



def defects_analysis(input_folder,
                     output_folder,
                     save_figures = True):

    # Scan the input folder and sort the images.
    image_list = glob.glob(input_folder + "/*.tif")
    image_list = natsorted(image_list, key=lambda y: y.lower())

    im_num = 0
    fig, ax1 = plt.subplots(1,1,  figsize=(8,8))

    ax1.clear(); ax1.axis('off') 

    # Retrieve the image dimensions.
    img_test = cv2.imread(image_list[im_num])[:,:,0]
    sizeX = img_test.shape[0]
    sizeY = img_test.shape[0]
    del img_test
    
    imgR = cv2.imread(image_list[im_num])[:,:sizeX,0]
    imgL = cv2.imread(image_list[im_num])[:,sizeX:,0]
    y, x = np.mgrid[0:imgR.shape[0], 0:imgR.shape[1]]

    sigma = 11
    oriR, plusR, minR = analyze_defects(imgR, sigma=sigma)
    oriL, plusL, minL = analyze_defects(imgL, sigma=sigma)
    s = 20
    ax1.imshow(np.zeros_like(imgR, dtype=np.float32), cmap="gray")
    for color,alpha, ori in zip(["c","m"],[1,.5], [oriL,oriR]):
        ax1.quiver(x[::s,::s], y[::s,::s],
                   np.cos(ori)[::s,::s], np.sin(ori)[::s,::s], 
                   headaxislength=0, headwidth=0, headlength=0, width=.01, 
                   color=color, scale=35, pivot='mid', alpha=alpha)
    
    plt.gca().set_box_aspect(1)

    if save_figures:
        fig.savefig(output_folder + "/LR direction quiver.svg")

    # Clean up.
    del imgR
    del imgL



def plot_velocity_field(input_folder,
                        output_folder,
                        save_figures = True):

    # Scan the input folder and sort the images.
    image_list = glob.glob(input_folder + "/*.tif")
    image_list = natsorted(image_list, key=lambda y: y.lower())

    im_num = 0
    fig, ax1 = plt.subplots(1,1,  figsize=(8,8))

    s = 25

    # Retrieve the image dimensions.
    img_test = cv2.imread(image_list[im_num])[:,:,0]
    sizeX = img_test.shape[0]
    sizeY = img_test.shape[0]
    del img_test

    imgR1 = cv2.imread(image_list[im_num])[:,:sizeX,0]
    imgL1 = cv2.imread(image_list[im_num])[:,sizeX:,0]
    imgR2 = cv2.imread(image_list[im_num+1])[:,:sizeX,0]
    imgL2 = cv2.imread(image_list[im_num+1])[:,sizeX:,0]

    ax1.clear(); ax1.axis('off') 
    ax1.imshow(np.zeros_like(imgR1, dtype=np.float32), cmap="gray")

    for color, img1,img2 in zip(["r","g"],[imgL1,imgR1],[imgL2,imgR2]):
        flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2,
                                            flags=0) 
        flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=15)
        flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=15)
    
        y, x = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]    

        ax1.quiver(x[::s, ::s], y[::s, ::s], 
                   flow[::s, ::s, 0], -flow[::s, ::s, 1], 
                   color=color, scale=50, alpha=.8,
                   width=.005, minlength=0)  
    
    plt.gca().set_box_aspect(1) 

    if save_figures:
        fig.savefig(output_folder + "/LR velocity quiver.svg")


    # This block is commented because there aren't enough images
    # in the input folder for this calculation.
    # n, dn = 10, 3
    # flows = time_average(image_list[n:n+dn], image_list[n+1:n+dn+1])
    
    # for i in range(3):
    #     print(flows[i][:,:,0].mean(), flows[i][:,:,1].mean())

    # np.concatenate(flows, axis=-1)[:,:,::2].mean(axis=-1)
    # np.concatenate(flows, axis=-1)[:,:,1::2].mean(axis=-1)
    

    # Clean up.
    del imgR1
    del imgL1
    del imgR2
    del imgL2


def compute_okubo_weiss_field(input_folder,
                              output_folder,
                              save_figures = True):
    
    # Scan the input folder and sort the images.
    image_list = glob.glob(input_folder + "/*.tif")
    image_list = natsorted(image_list, key=lambda y: y.lower())
    
    sigma = 35
    im_num = 0
    img1 = cv2.imread(image_list[im_num])[:,900:,0]
    # img2 = cv2.imread(image_list[im_num+1])[:,900:,0]
    # flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 
    #     winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0) 

    n, dn = 0, 2
    flow = time_average(image_list[n:n+dn], image_list[n+1:n+dn+1])

    field = np.stack((flow[:,:,0], flow[:,:,1]), axis=-1)
    vorticity = gaussian_filter(curl_npgrad(field), sigma=sigma)
    vort_min = 0.005
    vortTFr = vorticity > vort_min
    vortTFl = vorticity < vort_min

    y, x = np.mgrid[0:img1.shape[0], 0:img1.shape[1]] 

    # ax1.axis('off') 
    u = gaussian_filter(flow[...,0], sigma=sigma)
    v = gaussian_filter(flow[...,1], sigma=sigma)

    # Calculate the Okubo-Weiss parameter
    ow_field = calculate_okubo_weiss(u, v)
    # ow_field = gaussian_filter(ow_field, sigma=25)

    # Set a threshold value to control vortex detection sensitivity
    threshold_value = 6.*np.median(ow_field)  # Adjust as needed
    vortex_mask = ow_field > threshold_value

    fig, ax1 = plt.subplots(1,1, figsize=(img1.shape[1]//110, img1.shape[0]//110))
    step = 20
    # Plot the detected vortices
    # c1 = ax1.imshow(ow_field, cmap='jet')
    # c2 = ax1.imshow(vortex_mask, cmap='gray')#, alpha=.6)
    # ax1.quiver(x[::step, ::step], y[::step, ::step], 
    #         u[::step, ::step], -v[::step, ::step], 
    #         color="k", scale=60, alpha=.9, width=.003, minlength=0)
    flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=15)
    flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=15)

    vort_min = 0.005
    vortTFr = vorticity > vort_min
    vortTFl = vorticity < vort_min
    ur = flow[:,:,0] * vortTFr
    vr = flow[:,:,1] * vortTFr
    ul = flow[:,:,0] * vortTFl
    vl = flow[:,:,1] * vortTFl

    ax1.quiver(x[::step, ::step], y[::step, ::step], 
               ur[::step, ::step], -vr[::step, ::step], 
               color="b", scale=70, alpha=.8, width=.004, minlength=0)
    ax1.quiver(x[::step, ::step], y[::step, ::step], 
               ul[::step, ::step], -vl[::step, ::step], 
               color="r", scale=70, alpha=.8, width=.004, minlength=0)  

    mm = (vortex_mask*vortTFr) - 1*(vortex_mask*vortTFl)
    cmp = ListedColormap(['red', 'white', 'blue'])
    c2 =ax1.imshow(mm, cmap=cmp, alpha=.6)
    ax1.set_axis_off()


    # labeled_image, num_features = label(vortex_mask)
    # regions = regionprops(labeled_image)
    # areas = []
    # for i,props in enumerate(regions):
    #     areas.append(props.area)
    #     yi, xi = props.centroid
    #     ax1.text(xi,yi,"%.0f" % (props.area* (150/900)**2), color="k", fontsize=14)
    # plt.imshow(labels, cmap="tab10")
    # plt.title('Detected Vortices %.10s' %threshold_value)
    # fig.colorbar(c2)

    if save_figures:
        fig.savefig(output_folder + "/OW.svg")
    

