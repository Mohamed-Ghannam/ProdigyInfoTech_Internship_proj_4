# Cats VS. Dogs Image Classification Project

## Project Overview
This project aims to classify images of cats and dogs using a pre-trained deep learning model (VGG16) for feature extraction and Support Vector Classifier (SVC) for classification. The project involves loading and preprocessing image data, extracting features using the VGG16 model, training an SVC model, and evaluating the performance using various metrics.

## Libraries and Dependencies
The following libraries were used in this project:
- `os` for directory and file management
- `cv2` (OpenCV) for image loading and preprocessing
- `numpy` for numerical operations
- `matplotlib` for data visualization
- `sklearn` for model training and evaluation (specifically `SVC` and `classification_report`)
- `tensorflow` for loading the VGG16 pre-trained model
- `joblib` for saving and loading the trained SVC model

## Project Workflow
1. **Importing Libraries**: 
   Imported all necessary libraries including `os`, `cv2`, `numpy`, `matplotlib`, `sklearn`, `tensorflow`, and `joblib`.

2. **Loading the VGG16 Model**: 
   Loaded the VGG16 model (without the top layers) pre-trained on ImageNet to extract features from the images.

3. **Defining Functions**:
   - Function to display a sample of training images (cats and dogs).
   - Function to load images and their labels.
   - `extract_vgg_features` function to extract VGG16 features from the images.

4. **Data Loading and Preprocessing**:
   - Loaded training and testing data (unlabeled test data).
   - Preprocessed the images and extracted features using the VGG16 model.

5. **Model Training**:
   - Fitted a Support Vector Classifier (SVC) on the extracted features to classify images into cats or dogs.

6. **Evaluation**:
   - Printed a classification report to evaluate the performance of the model on the test data.

7. **Prediction**:
   - Predicted the labels of some unlabeled test images using the trained SVC model and a custom prediction function.

## Results
- The classification report provides insights into the precision, recall, F1-score, and accuracy of the model for classifying cats and dogs.

## Conclusion
This project successfully demonstrates the use of transfer learning with VGG16 for feature extraction and SVC for image classification. The model can accurately classify images of cats and dogs, with evaluation metrics showing good performance.

## Directory Structure
- `train/`: Contains training images for cats and dogs.
- `test/`: Contains unlabeled test images.
- `models/`: Folder for saving trained models.
