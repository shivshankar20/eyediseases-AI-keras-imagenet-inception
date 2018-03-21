# eyediseases-AI-keras-imagenet-inception
Image based deep learning to detect eye diseases. Transfer learning and feature extraction using keras, imagenet, and Inception v3.

There are two approaches covered here:

I. Transfer Learning Approach (94% accuracy, 100 hours of training duration, 500 epochs, 12 mins/epoch)

II. Feature Extraction & Bottleneck Approach (99.1% accuracy, 75 mins of training duration, 50 epochs, 90 sec/epoch)

## I. Transfer Learning Approach
Load a pretrained InceptionV3 model without the top layers, lock the base model, add new layers, and train the model.

Please read the following article before downloading code samples:

[Helping Eye Doctors to see better with machine learning (AI)](http://blog.mapshalli.org/index.php/2018/03/17/helping-eye-doctors-to-see-better-with-machine-learning-ai/)

### Jupyter notebooks
* Train.ipynb - Train the model
* Evaluate.ipynb - Evaluate the model

### Models

See the documentation below to know how to download the trained model (84MB!)


### Documentation
[Helping Eye Doctors to see better with machine learning (AI)](http://blog.mapshalli.org/index.php/2018/03/17/helping-eye-doctors-to-see-better-with-machine-learning-ai/)

The article also consists of link to download saved Keras model with weights.

### Table of Contents

#### Part 1 – Background
* Introduction
* Optical coherence tomography (OCT) and eye diseases
* Normal Eye Retina (NORMAL)
* Choroidal neovascularization (CNV)
* Diabetic Macular Edema (DME)
* Drusen (DRUSEN)
* Teaching humans to interpret OCT images for eye diseases
* Teaching computers to interpret OCT images for eye diseases – Algorithmic Approach
* Teaching computers to interpret OCT images for eye diseases – Deep Neural Networks

#### Part 2 – Implementation: Train the model
* Introduction
* Selection and Installation of Deep Learning Hardware and Software
* Download the data and organize
* Import required Python modules
* Setup the training and test image data generators
* Load InceptionV3 and attach new layers at the top
* Compile the model
* Fit the model with data and save the best model during training
* Monitor the training and plot the results

#### Part 3 – Implementation: Evaluate the Model
* Introduction
* Import required Python modules
* Load the saved best model
* Evaluate the model for a small set of images
* Write utility functions to get predictions for one image at a time
* Implement grad_CAM function to create occlusion maps
* Make prediction for a single image and create an occlusion map
* Make predictions for multiple images and create occlusion map for misclassified images

#### Part 4: Summary and Download links


## II. Feature Extraction & Bottleneck Approach
Feed all the images (training and validation) to extract the output of the base InceptionV3 model.  Save the outputs, i.e, features (bottlenecks) and the associated labels in a file. Use a shallow neural network and feed the saved features to train the model. Save
the best performing model found during training and reduce the learning rate if the validation loss remains flat for few epochs.

Please read the following article before downloading code samples:

[Faster and better transfer learning training with deep neural networks (AI) to detect eye diseases](http://blog.mapshalli.org/index.php/2018/03/21/faster-and-better-transfer-learning-training-with-deep-neural-networks-ai-to-detect-eye-diseases/)

### Highlights

* Extract features by feeding the images to an InceptionV3 model trained with imagenet dataset.
* Save training and validation features to h5 files.
* Create your own small neural network model.
* Write generators to feed saved features to your model.
* Achieve 99.1% accuracy
* Training speed reduced to 1/10th. 12 mins/epoch to 1.5 min/epoch!


### Jupyter notebooks
* Features-Extract.ipynb - Extract features by feeding images through InceptionV3 pretrained model
* Features-Train.ipynb - Feed the features to train a small neural network with a classifier
* Features-Evaluate.ipynb - Evaluate the model along with occlusion maps

### Models
output/model.24-0.99.hdf5.zip:
* Model file created by Features-Train.ipynb.
* Unzip and load the model in Features-Evaluate.ipynb

### Documentation
[Faster and better transfer learning training with deep neural networks (AI) to detect eye diseases](http://blog.mapshalli.org/index.php/2018/03/21/faster-and-better-transfer-learning-training-with-deep-neural-networks-ai-to-detect-eye-diseases/)

### Table of Contents

#### Part 1 – Background and Overview
* Transfer learning – using a fully trained model as a whole
* Transfer learning – extract features (bottlenecks), save them and feed to a shallow neural network

#### Part 2 – Implementation
* Extract features using imagenet trained InceptionV3 model
  * Import the required modules and load the InceptionV3 model
  * Extract features by feeding images and save the features to a file
* Build and train a shallow neural network
  * Import the required modules
  * Setup a generator to feed saved features to the model
  * Build a shallow neural network model
  * Train the model, save the best model and tune the learning rate
* Evaluate the model
  * Import the required modules and load the saved model
  * Evaluate the model by making predictions and viewing the occlusion maps for multiple images

#### Part 3 – Summary and Download links
