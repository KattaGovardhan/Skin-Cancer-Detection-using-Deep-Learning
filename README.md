# Skin-Cancer-Detection-using-Deep-Learning

## Description

The Skin Cancer Detection and Classification project is a machine learning application designed to identify and classify various types of skin cancer from images using Convolutional Neural Networks (CNN). The application uses TensorFlow and Keras for model building and training, and Tkinter for creating a user-friendly graphical interface. It supports dataset loading, preprocessing, model training, prediction, and visualization of results.

## Features

- Upload and preprocess ISIC skin cancer dataset.
- Build and train a CNN model for skin cancer classification.
- Predict skin cancer from test images.
- Display model accuracy, loss, and confusion matrix.
- Visualize model performance through accuracy and loss graphs.

## Installation Instructions

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

   Make sure `requirements.txt` includes:
    ```plaintext
    numpy
    matplotlib
    opencv-python
    tensorflow
    scikit-learn
    seaborn
    ```

4. Download the ISIC skin cancer dataset and place it in the appropriate directory.

## Usage Instructions

1. Run the application:
    ```bash
    python <your-script-name>.py
    ```

2. Use the following buttons in the GUI:
    - **Upload ISIC Dataset**: Load the dataset directory.
    - **Preprocess Dataset**: Process the images and labels for training.
    - **Build SCDC Model**: Train the CNN model or load an existing model.
    - **Upload Test Data & Predict Disease**: Select an image to predict the skin disease.
    - **Accuracy Comparison Graph**: Display a graph of the model's accuracy and loss over epochs.

## Dependencies

- Python 3.x
- Tkinter
- NumPy
- Matplotlib
- OpenCV
- TensorFlow
- scikit-learn
- Seaborn

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your proposed changes.

## Author

Govardhan Katta


## Known Issues

- Ensure that the paths in the code match the locations of your dataset and model files.
- The application might require adjustments based on your specific dataset and environment.

---

Feel free to modify this template as needed and add any additional information or images that might be relevant.
