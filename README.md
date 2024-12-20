# Handwritten Digit Classification Using ANN

This project demonstrates a simple approach to classifying handwritten digits using an Artificial Neural Network (ANN) with the MNIST dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

In this project, we build a neural network model to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). The goal of this project is to build an ANN model to classify these digits based on the images provided.

The architecture of the model consists of an input layer, a hidden layer, and an output layer with 10 neurons representing the digits 0-9. We use the **Keras** library for building the model, and **TensorFlow** as the backend.

## Dataset

The dataset used in this project is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which is a popular dataset for training image classification models. It contains:

- 60,000 training images
- 10,000 test images

Each image is 28x28 pixels and represents a handwritten digit from 0 to 9.

## Libraries Used

This project requires the following Python libraries:

- **TensorFlow**: To build and train the neural network model.
- **Keras**: A high-level neural networks API that runs on top of TensorFlow.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing the results.
- **Pandas**: For data manipulation (optional).
- **Scikit-learn**: For data preprocessing (optional).

### To install the required libraries, run:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas
