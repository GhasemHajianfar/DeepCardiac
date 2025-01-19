# Saliency Map Generator for Cardiac Analysis

This repository contains a script to generate saliency maps using a pre-trained deep learning model for cardiac analysis. The script loads input data, applies necessary preprocessing, runs inference, and visualizes the results with a Grad-CAM saliency map.

## Features

- Loads a pre-trained model and its weights.
- Processes numerical and image data.
- Generates saliency maps using Grad-CAM.
- Outputs prediction probabilities and labels.
- Saves visualizations as high-resolution images.

## Requirements

- Python 3.9+
- TensorFlow 2.14
- Pandas
- h5py
- pickle
- Matplotlib
- skimage
- cv2

Install the required packages:

```sh
pip install tensorflow pandas h5py matplotlib skimage
```

## Project Structure

```
.
├── img_array.h5           # Input image array in HDF5 format
├── numerical_feature.xlsx # Input numerical features in Excel format
├── zscore.pkl             # Pre-trained z-score transformer (Pickle format)
├── threshold.xlsx         # Threshold dictionary (Excel format)
├── model.hdf5             # Pre-trained model file
├── model_weight.h5        # Model weights file
├── Silency_map.jpg        # Output saliency map (generated)
├── inference_script.py    # Main script to run inference
└── utilit.py              # Utility functions (e.g., make_saliency_map, save_and_display_gradcam)
```

## Usage

### Running the Script

The script assumes all required files are located in the same directory as the script. To run the script, use:

```sh
python3 inference_script.py
```

### Input Files

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| img_array.h5           | HDF5 file containing image array data.           |
| numerical_feature.xlsx | Excel file containing numerical features.        |
| zscore.pkl             | Pickle file containing a pre-trained z-score transformer. |
| threshold.xlsx         | Excel file containing thresholds.                |
| model.hdf5             | HDF5 file of the pre-trained model.              |
| model_weight.h5        | File containing weights of the pre-trained model.|

### Output Files

| File            | Description                                      |
|-----------------|--------------------------------------------------|
| Silency_map.jpg | Generated saliency map saved as a JPG image.     |

### Output

- **Prediction Probability:** Displays the prediction probabilities for each label.
- **Prediction Label:** Displays the predicted labels for the input data.
- **Saliency Map:** Saves a saliency map visualization as `Silency_map.jpg`.

### Example Workflow

1. Prepare the input files (`img_array.h5`, `numerical_feature.xlsx`, `zscore.pkl`, etc.) in the same directory as the script.
2. Run the script using `python3 inference_script.py`.
3. View the outputs in the terminal and the generated saliency map at `Silency_map.jpg`.
