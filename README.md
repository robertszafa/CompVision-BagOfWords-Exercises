# Introduction

SIFT function implementation for feature detection from an image. Our example classifies images for the following classes: airplanes, cars, dog, faces and keyboard.

We use the Bag of Word model where we generate codewords for each class, i.e. 20,000 features from the dog class. Once we generated the codewords for each classes, we can create a dictionary of 500 codeword and normalise it. Once we have 500 codewords we can use it to create a histogram for all the test and training images. The histogram bin represents a codeword/feature that will be used to help us classify or label the image based off the classes we mentioned.
# How to run

How to run the different steps of the Bag of Word model (and the time it takes). Paste the python commands into console/terminal.

Tested on machine with following specs:
1. 10th Generation Intel Core i7-1065G7, Windows 10 Home 64bit, 16 GB RAM, Intel Iris Plus Graphics

## Step 1 - SIFT Descriptor

* Extract SIFT descriptors from training and test images
* Stores as binary file ***...descriptors.npy***
* Takes X hours

``` 
python SIFT.py
```

## Step 2 - Generate Codebook

* Generate the codebook descriptors by euclidean and sad
* Also generate the smaller codebook with cluster of 20
* Stores as binary file ***.npy***

``` 
python gen_codebook.py
```