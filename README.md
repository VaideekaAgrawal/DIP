# 🎨 Image Classification: CG vs. Camera Images (Google Colab)

This repository contains a Google Colab notebook that implements the image classification task from the paper:  
**"Forensic Techniques for Classifying Scanner, Computer Generated and Digital Camera Images."**  

The goal is to classify images as **Computer-Generated (CG)** or **Camera-Captured** by extracting 45 features (15 per RGB channel) and training an **SVM classifier**.

---

## 📂 Project Structure

├── /notebooks/
│   └── CG_vs_Camera_Classification.ipynb
├── /data/
│   ├── DIP_CG/             # Folder with CG images
│   └── DIP_DigitalCamera/  # Folder with Camera images
├── /results/
│   ├── cropped_images/     # Cropped images
│   ├── feature_vectors.csv # Extracted features
│   ├── feature_vectors.npz # Numpy feature vectors
│   └── svm_model.joblib    # Trained SVM model
├── README.md
└── requirements.txt
---

## 🚀 Features

✅ Extracts 15 noise-based features per channel (R, G, B) from images  
✅ Crops/resizes images to standard dimensions (1024x768)  
✅ Uses wavelet denoising for noise estimation  
✅ Trains an **SVM** model with hyperparameter tuning (GridSearchCV)  
✅ Generates a confusion matrix and classification report  
✅ Saves model and feature vectors for reproducibility  

---

## 🏗️ Installation

Clone the repository and install the dependencies in your Google Colab notebook:

```bash
!pip install opencv-python-headless scikit-image scikit-learn matplotlib Pillow seaborn PyWavelets pdf2image
!apt-get update
!apt-get install -y poppler-utils

## 🚀 Features

✅ Extracts 15 noise-based features per channel (R, G, B) from images
✅ Crops/resizes images to standard dimensions (1024x768)
✅ Uses wavelet denoising for noise estimation
✅ Trains an SVM model with hyperparameter tuning (GridSearchCV)
✅ Generates a confusion matrix and classification report
✅ Saves model and feature vectors for reproducibility

## 🧮 Feature Extraction

For each image, 45 features are extracted:
	•	Per channel (15 features):
	•	Mean, std, skew, kurtosis of row/column-wise correlations (rho_row, rho_col)
	•	Std, skew, kurtosis of row/column averages (r_avg, c_avg)
	•	Ratio feature: (1 - mean(rho_col) / mean(rho_row)) * 100

⸻

📦 Outputs
	•	📁 cropped_images/ – Preprocessed images (center-cropped & resized)
	•	📄 feature_vectors.csv – Extracted features + labels + filenames
	•	📦 feature_vectors.npz – Numpy arrays of features, labels, and filenames
	•	💾 svm_model.joblib – Trained SVM model
	•	📊 Confusion matrix and classification report


⸻

🏃 Usage

1️⃣ Extract features and preprocess images:
  X, y, filenames = load_dataset(cg_folder, camera_folder)
2️⃣ Train the SVM model:
  grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
3️⃣ Save the model:
  joblib.dump(clf, "svm_model.joblib")
4️⃣ Evaluate:
  y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

📈 Results
	•	Example confusion matrix and classification report are generated during training.
	•	Model performance depends on the dataset quality and size.

⸻

📚 References
	•	Forensic Techniques for Classifying Scanner, Computer Generated and Digital Camera Images (Paper)
	•	Wavelet Denoising: skimage.restoration.denoise_wavelet
	•	SVM Implementation: scikit-learn
