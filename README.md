# RETINARES: An integrated deep learning framework for Diabetic Retinopathy classification and Real-Time Severity Preiction 
Diabetic Retinopathy (DR) is a severe eye condition that affects individuals with diabetes, potentially leading to blindness if left untreated. This project aims to leverage deep learning techniques to develop an automated system for detecting DR from retinal images. The system is built using a ResNet-50 model,to enhance feature extraction, improve classification accuracy an to detect subtle features associated with DR severity.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and can lead to blindness if not detected early. This project aims to automate the detection and classification of DR using deep learning techniques. The model classifies images into five categories:

- **No_DR**: No signs of diabetic retinopathy
- **Mild**: Early signs of the disease
- **Moderate**: More significant signs without severe damage
- **Severe**: Severe damage indicating significant disease progression
- **Proliferate_DR**: Advanced stage requiring immediate medical attention

## Features

- **Deep Learning-Based Classification**: Utilizes ResNet-50 architecture.
- **Web Interface**: A Flask-based web application allows for easy image upload and DR prediction.
- **Performance Tracking**: Detailed accuracy, precision, recall, and F1 score metrics.
- **GPU Support**: Optimized for training on machines with GPUs.

## Dataset
The dataset used in this project is a collection of retinal images with varying stages of diabetic retinopathy. The dataset is pre-processed and split into training, validation, and test sets.

**Source**: Kaggle Diabetic Retinopathy Dataset

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.8
- TensorFlow 2.x

### Step 1: Clone the Repository

```bash
git clone https://github.com/YourUsername/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection 
```
### Step 2: Create a Conda Environment

```bash
conda create -n drdetection python=3.8
conda activate drdetection
````
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Set Up the Dataset
Ensure your dataset is in the correct format and directory structure as required by the project. Update paths in the code if necessary.

## Usage
### Step 1: Train the Model
You can train the model using the provided Jupyter notebook or directly through Python scripts.

```bash
jupyter notebook
```
### Step 2: Run the Flask Application
To start the web interface:

```bash
python app.py
```
Visit http://127.0.0.1:5000/ in your browser to upload an image and predict DR severity.

### Step 3: Deactivate the Environment
```bash
conda deactivate
```

## Model Architecture
The model is based on the ResNet50 architecture, with self-attention layers added after each convolutional layer to improve focus on relevant features within retinal images. The architecture includes the following components:

- ResNet50 Backbone: Pre-trained on ImageNet for initial feature extraction.
- Dense Layers: For classification into the five DR severity categories.
- Batch Normalization and Dropout: Used for regularization to prevent overfitting.
![image](https://github.com/user-attachments/assets/45c2bbcb-77bb-4137-8c5c-66a1fc5999cb)

## Results
- Homepage

  
  ![image](https://github.com/user-attachments/assets/32b3328c-b3eb-4db3-bb7a-d86e2fdf5c2c)
  
- Predcition Page


  ![image](https://github.com/user-attachments/assets/02b6470d-3155-42f8-89ad-cbe84654d999)

- Prediction Result


  ![image](https://github.com/user-attachments/assets/0fded444-4937-474f-a769-f22231f9951e)


## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Create a new Pull Request.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
