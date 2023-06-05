# CELEB-IO
> Created By: Audrey Craig, Lianne Kniest and Shreya Ramakrishnan


CELEB-IO is a machine learning model developed using Keras and SVM (Support Vector Machines). It is designed to classify images of celebrities and predict the identity of each celebrity in a given dataset. In addition to the core model, the project includes a webcam extension that provides real-time celebrity lookalike identification. The summary below highlights the key aspects of the project. 

## Summary 

### Problem Setup 

The goal of CELEB-IO is to train a model to accurately predict the identity of celebrities depicted in images. It involves multi-class classification, where each celebrity corresponds to a separate class label. We wanted to build an accurate facial recognition model and knowing that images of celebrities are very wide-spread across the internet, we knew that would be a robust dataset to train a facial recognition model on. Additionally, wanting to add a real time component to our project we chose to implement the webcam extension. 

When approaching the project, we knew that the breakdown of the project would look similar to the following.
1. Build a model using pre-trained weights. We decided to use a CNN (Convolutional Neural Network) as they are the most accurate at image classification. We drew inspiration from the VGG face model and looked through the documentation to see the different network layers they used to replicate a similar structure. 
2. Generate embeddings for all the images in our dataset using the CNN that we built earlier. These image embeddings will then be processed and passed into an SVM classifier that we build later. 
3. Process the embeddings using PCA into the appropriate format to fit the SVM classifier we build in the next step. 
4. Create and fit an SVM classifier. Additionally, we use the Label Encoder class to create a Label Encoder that we can use to output names rather than their numerical encodings. We create the LE in this step as well, and use both it and the SVM classifier to output predictions on our test set. 
5. Train the model on a much larger data set that we curated (dataset creation will be expanded on in the following sections). 
6. Implement the webcam extension using Open CV2 and call the model on frames extracted from the webcam. 

### Data Collection and Preprocessing 
> DATASET LINK: https://drive.google.com/drive/folders/0B5G8pYUQMNZnLTBVaENWUWdzR0E?resourcekey=0-gRGzioHdCR4zkegs6t1W2Q&usp=sharing
> Given by https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset

We selected this dataset because of its vast array of celebrities including those of different races, gender, and age. The robust dataset had also been used for a similar project and was therefore organized into celebrity-named folders, eliminating extra preprocessing steps. In terms of preprocessing the data, we wrote a script (contained in face_crop.py) which iterated through the images and extracted the face. We also implemented checks that threw out data that had more than one face in the image. This was because each image was tagged with one celebrity name, and an image that contained two faces may confuse the model. We also discarded any images that did not have any identifiable faces. Once this processing was complete, the processed images were saved into a new folder which would be used for the model. Images in this folder followed the naming convention "Celebrity Name_Number.png", where number represents 1 for the first image of that celebrity, 2 for the second, and so on. 

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
 
 
  
