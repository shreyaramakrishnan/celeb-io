# CELEB-IO
> Created By: Audrey Craig, Lianne Kniest and Shreya Ramakrishnan


CELEB-IO is a machine learning model developed using Keras and SVM (Support Vector Machines). It is designed to classify images of celebrities and predict the identity of each celebrity in a given dataset. In addition to the core model, the project includes a webcam extension that provides real-time celebrity lookalike identification. The summary below highlights the key aspects of the project. 

## Summary 

### Problem Setup 

The goal of CELEB-IO is to train a model to accurately predict the identity of celebrities depicted in images. It involves multi-class classification, where each celebrity corresponds to a separate class label. We wanted to build an accurate facial recognition model and knowing that images of celebrities are very wide-spread across the internet, we knew that would be a robust dataset to train a facial recognition model on. Additionally, wanting to add a real time component to our project we chose to implement the webcam extension. 

When approaching the project, we knew that the breakdown of the project would look similar to the following.
1. Build a model using pre-trained weights. We decided to use a CNN (Convolutional Neural Network) as they are the most accurate at image classification. We drew inspiration from the VGG face model and looked through the documentation to see the different network layers they used to replicate a similar structure. 
2. Generate embeddings for all the images in our dataset using the CNN that we built earlier. These image embeddings will then be processed and passed into an SVM classifier that we build later. 
3. Process the embeddings  

### Data Collection and Preprocessing 

### Pre-existing Components 

### Original Implementation 



Project Video Summary Link:

1-2 paragraphs about project:



      What problem are you trying to solve?

      What algorithms/techniques will you use?

      What dataset(s)?

      Any other useful information
      
      
      
 This video summary should mention...
 
 setup:
 
 data set used: https://drive.google.com/drive/folders/0B5G8pYUQMNZnLTBVaENWUWdzR0E?resourcekey=0-gRGzioHdCR4zkegs6t1W2Q&usp=sharing
 given by https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset
 
 techniques:
 
 components from preexisting work: VGG Face model inspired by https://www.kaggle.com/general/255813
 
 components implemented for the project: 
 
 
  
