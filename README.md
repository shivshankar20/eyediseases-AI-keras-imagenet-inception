# eyediseases-AI-keras-imagenet-inception
Image based deep learning to detect eye diseases. Transfer learning using keras, imagenet, and Inception v3.

## I. Transfer Learning Approach
Load a pretrained InceptionV3 model without the top layers, lock the base model, add new layers, and train the model.

Please read the following article before downloading code samples:

[Helping Eye Doctors to see better with machine learning (AI)](http://blog.mapshalli.org/index.php/2018/03/17/helping-eye-doctors-to-see-better-with-machine-learning-ai/)

### Jupyter notebooks
* Train.ipynb - Train the model
* Evaluate.ipynb - Evaluate the model


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


## II. Bottleneck Approach
Coming soon! 
* Extract features by feeding the images to an InceptionV3 model trained with imagenet dataset.
* Save features in a h5 file.
* Create your own small neural network model.
* Write generators to feed saved features to your model
* Achieve 98% accuracy
* Training speed reduced to 1/10th. 10 mins/epoch to 1 min/epoch!
