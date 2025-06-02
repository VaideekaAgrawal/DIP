# Image Classification: CG vs. Camera

This repository contains a Google Colab notebook implementing an image classification task to distinguish between Computer Generated (CG) and Camera-captured images, based on the methodology from the paper *"Forensic Techniques for Classifying Scanner, Computer Generated and Digital Camera Images"*. The implementation uses 45 features extracted from the RGB channels of images to train a Support Vector Machine (SVM) model with an RBF kernel.

## Features
- **Image Preprocessing**: Images are center-cropped to a fixed resolution (1024x768) using PIL.
- **Feature Extraction**: Extracts 15 statistical features per RGB channel (45 total) using wavelet denoising, noise analysis, and correlation metrics.
- **Model Training**: Trains an SVM classifier with grid search for optimal hyperparameters (C, gamma).
- **Output**: Saves the trained model, feature vectors, and confusion matrix to specified directories.
- **PDF Processing**: Includes functionality to extract features from the first page of a PDF file.

## Requirements
The notebook requires the following Python libraries and tools, which are installed in the Colab environment:
```bash
pip install opencv-python-headless scikit-image scikit-learn matplotlib Pillow seaborn PyWavelets pdf2image
apt-get install poppler-utils
```

## Directory Structure
- **Input**:
  - `DIP_CG/`: Folder containing Computer Generated images.
  - `DIP_DigitalCamera/`: Folder containing Camera-captured images.
- **Output**:
  - `DIP_Output/cropped_images/`: Stores center-cropped images.
  - `DIP_Output/results/`: Stores the trained SVM model (`svm_model.joblib`), feature vectors (`feature_vectors.csv`, `feature_vectors.npz`), and confusion matrix (`confusion_matrix.png`).

## Usage
1. **Set Up Google Drive**:
   - Mount your Google Drive in Colab to access input images and save outputs.
   - Ensure the `DIP_CG` and `DIP_DigitalCamera` folders are in `/content/drive/MyDrive/`.

2. **Run the Notebook**:
   - Execute the cells in the provided Colab notebook.
   - The notebook will:
     - Install required libraries.
     - Preprocess images (crop/resize).
     - Extract features from RGB channels.
     - Train an SVM model with grid search.
     - Save the model, feature vectors, and confusion matrix.
     - Optionally process a PDF file to extract features.

3. **PDF Processing**:
   - Provide a PDF file path to `process_pdf_for_features()` to extract features from the first page.

## Code Overview
- **center_crop_image(filepath, output_path)**: Crops images to 1024x768 resolution, saving the result if an output path is provided.
- **extract_features(img)**: Extracts 15 features per RGB channel, including statistical measures (mean, std, skew, kurtosis) of noise correlations and row/column averages.
- **load_dataset(cg_folder, camera_folder)**: Loads and processes images from CG and Camera folders, saving cropped images and returning feature vectors, labels, and filenames.
- **SVM Training**: Uses `GridSearchCV` to tune SVM hyperparameters and trains the model. The best model is saved as `svm_model.joblib`.
- **Output Generation**: Saves feature vectors as CSV and NPZ files, and generates a confusion matrix visualization.
- **PDF Feature Extraction**: Converts the first page of a PDF to an image and extracts features using `extract_features()`.

## Example Output
- **Feature Vectors**: Saved as `feature_vectors.csv` and `feature_vectors.npz` in the results directory.
- **Trained Model**: Saved as `svm_model.joblib`.
- **Confusion Matrix**: Visualized and saved as `confusion_matrix.png`.
- **PDF Features**: Example output for a PDF file:
  ```python
  Extracted features: [-4.63989347e-02, 2.65201083e-01, 9.10990467e-02, ...]
  ```

## Performance
The SVM model achieves approximately 81% accuracy on the test set, with detailed metrics provided in the classification report:
```
Classification Report:
               precision    recall  f1-score   support
           0       0.80      0.82      0.81        49
           1       0.82      0.80      0.81        51
    accuracy                           0.81       100
   macro avg       0.81      0.81      0.81       100
weighted avg       0.81      0.81      0.81       100
```

## Notes
- **Error Handling**: The code includes error handling for corrupted or unreadable image files, with logs printed for problematic files.
- **NaN Handling**: Uses `SimpleImputer` to handle NaN values in feature vectors with mean imputation.
- **Scalability**: Ensure sufficient storage in Google Drive for cropped images and results.
- **Dependencies**: The notebook is designed for Google Colab with specific library versions (e.g., PyWavelets 1.8.0, numpy 2.0.2).

## Citation
This implementation is based on the methodology described in:
- *Forensic Techniques for Classifying Scanner, Computer Generated and Digital Camera Images* (please refer to the original paper for full citation details).

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.