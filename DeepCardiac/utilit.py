#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:15:58 2025

@author: ghasem
"""
import os,glob
import numpy as np # linear algebra
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving figures
def make_saliency_map(img_array, nf ,img_array_tf,nf_tf,thresholds_dict, model, pred_index=None):
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the input image
    with tf.GradientTape(persistent=True) as tape:
            tape.watch(img_array_tf)
            predictions = model([img_array_tf, nf])
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]
            predictions2=predictions.numpy()
            y_pred = np.zeros_like(predictions)
            for i, target_name in enumerate(thresholds_dict.keys()):
                print(target_name)
                threshold = thresholds_dict[target_name]
                y_pred[0,i] = (predictions2[0,i]>= threshold).astype(int)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the input image
    gradient = tape.gradient(class_score, img_array_tf)
    # Rectify gradients using ReLU
    gradient = tf.nn.relu(gradient)
    # Take the absolute value of the gradients for visualization
    saliency_map = tf.abs(gradient[0])
    #saliency_map=gradient[0]
    # Normalize the saliency map between 0 and 1
    max_value = tf.reduce_max(saliency_map)
    if max_value != 0:
        saliency_map /= max_value
    return saliency_map.numpy(), predictions.numpy(),y_pred

def fill_holes(label_image):
    # Convert label image to uint8
    label_image_uint8 = np.uint8(label_image)

    # Perform morphological dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(label_image_uint8, kernel, iterations=1)

    # Find contours of dilated image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill inside contours in original label image
    filled_image = np.zeros_like(label_image)
    for contour in contours:
        cv2.drawContours(filled_image, [contour], 0, 1, -1)

    return filled_image

def save_and_display_gradcam(img_array, heatmap,prob,pred, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img_rest=img_array[0,:,:,0]
    img_stress=img_array[0,:,:,1]
    heatmap_rest=heatmap[:,:,0]
    heatmap_stress=heatmap[:,:,1]
    #img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap_rest = np.uint8(255 * heatmap_rest)
    heatmap_stress = np.uint8(255 * heatmap_stress)
    mask_rest = (img_rest == 0)
    heatmap_rest[mask_rest] = 0
    mask_stress = (img_stress == 0)
    heatmap_stress[mask_stress] = 0

    thresh_rest = threshold_otsu(heatmap_rest)
    bw_rest = closing(heatmap_rest > thresh_rest, square(3))
    
    # label image regions
    label_image_rest = label(bw_rest)
    label_image_rest[label_image_rest>0]=1
    label_image_rest=fill_holes(label_image_rest)
    heatmap_rest=label_image_rest*heatmap_rest
    
    thresh_stress = threshold_otsu(heatmap_stress)
    bw_stress = closing(heatmap_stress > thresh_stress, square(3))
    
    # label image regions
    label_image_stress = label(bw_stress)
    label_image_stress[label_image_stress>0]=1
    label_image_stress=fill_holes(label_image_stress)
    heatmap_stress=label_image_stress*heatmap_stress
    
    # Mask out zero values
    jet_heatmap_rest = np.ma.masked_where(heatmap_rest == 0, heatmap_rest)
    jet_heatmap_stress = np.ma.masked_where(heatmap_stress == 0, heatmap_stress)    
    jet_heatmap_rest = np.ma.masked_where(heatmap_rest == 0, heatmap_rest)
    jet_heatmap_stress = np.ma.masked_where(heatmap_stress == 0, heatmap_stress)
    alpha=0.6
    # Superimpose the heatmap onto the original image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_rest, cmap="Greys", vmin = np.min(img_rest), vmax = np.max(img_rest))
    ax[0].imshow(jet_heatmap_rest, alpha=alpha, interpolation="none", cmap='jet', filternorm = False, resample = False)
    ax[0].axis('off')
    ax[0].set_title('Rest' , fontsize=20, fontweight='bold')

    ax[1].imshow(img_stress, cmap="Greys")
    ax[1].imshow(jet_heatmap_stress, alpha=alpha, interpolation="none", cmap='jet', filternorm = False, resample = False)
    ax[1].axis('off')
    ax[1].set_title('Stress', fontsize=20, fontweight='bold')
    
        # Add text annotations
    ax[0].text(1.1, 0.01, f" LAD: {prob[0][0]:.2f} \nRCA: {prob[0][1]:.2f}\nLCX: {prob[0][2]:.2f}", 
                   horizontalalignment='center', verticalalignment='bottom', 
                   transform=ax[0].transAxes, fontsize=15, color='black', weight='bold')
    ax[0].text(1.1, 0.99, f" LAD: {pred[0][0]:.2f} \nRCA: {pred[0][1]:.2f}\nLCX: {pred[0][2]:.2f}", 
                   horizontalalignment='center', verticalalignment='top', 
                   transform=ax[0].transAxes, fontsize=15, color='black', weight='bold')

    plt.tight_layout()
    plt.savefig(cam_path,
                dpi=600, facecolor='w', edgecolor='w', orientation='portrait', format='jpg',
                transparent=False, bbox_inches='tight', pad_inches=0.1)
    return label_image_stress ,label_image_rest

