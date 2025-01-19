
#!/usr/bin/env python3

import argparse
import h5py
import pickle
import pandas as pd
import tensorflow as tf
from utilit import make_saliency_map, fill_holes,save_and_display_gradcam

def main():
    # Default paths
    img_array_path = "img_array.h5"
    numerical_feature_path = "numerical_feature.xlsx"
    zscore_path = "zscore.pkl"
    thresholds_path = "threshold.xlsx"
    model_path = "model.hdf5"
    model_weights_path = "model_weight.h5"
    saliency_map_path = "Silency_map.jpg"
    
    # Load the HDF5 image array
    with h5py.File(img_array_path, "r") as h5f:
        img_array = h5f["img_array"][:]
    
    # Load numerical features and transform them
    nf = pd.read_excel(numerical_feature_path)
    with open(zscore_path, 'rb') as f:
        zscore = pickle.load(f)
    nf = zscore.transform(nf)
    
    # Load thresholds dictionary
    thresholds_dict = pd.read_excel(thresholds_path)
    thresholds_dict = thresholds_dict.set_index('Unnamed: 0')[0].to_dict()
    
    # Load model and weights
    model_base = tf.keras.models.load_model(model_path)
    model_base.load_weights(model_weights_path)
    
    # Convert arrays to tensors
    img_array_tf = tf.convert_to_tensor(img_array, dtype=tf.float32)
    nf_tf = tf.convert_to_tensor(nf, dtype=tf.float32)
    
    # Generate saliency map and predictions
    heatmap, prob, pred = make_saliency_map(img_array, nf,img_array_tf,nf_tf,thresholds_dict, model_base, pred_index=None)
    
    # Save and display Grad-CAM
    label_image_stress, label_image_rest = save_and_display_gradcam(
        img_array, heatmap, prob,pred,cam_path=saliency_map_path)
    
    # Print results
    print(f"Prediction Probability: {prob}")
    print(f"Prediction Label: {pred}")
    print(f"Saliency map saved at: {saliency_map_path}")

if __name__ == "__main__":
    main()