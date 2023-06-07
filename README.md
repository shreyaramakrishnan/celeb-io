# CELEB-IO
> Created By: Audrey Craig, Lianne Kniest and Shreya Ramakrishnan


CELEB-IO is a machine learning model developed using Keras and SVM (Support Vector Machines). It is designed to classify images of celebrities and trained by predicting the identity of each celebrity in a given dataset. In addition to the core model, the project includes a webcam extension that provides real-time celebrity lookalike identification, AKA "what celebrity do I most look like?". The summary below highlights the key aspects of the project. 

## Project Demo 
>[SEE DEMO HERE](https://www.loom.com/share/4428821655ac4c588ce6f67378a11022)

## Summary 

### Problem Setup 

The goal of CELEB-IO is to train a model to accurately predict the identity of celebrities depicted in images. It involves multi-class classification, where each celebrity corresponds to a separate class label, one class per celebrity. We wanted to build an accurate facial recognition model and knowing that images of celebrities are very wide-spread across the internet, this would require a robust dataset to train a facial recognition model on. Additionally, wanting to add a real time component to our project we chose to implement the webcam extension. 

When approaching the project, we knew that the breakdown of the project would look similar to the following.
1. Build a functional model for generating face embeddings using pre-trained weights. We decided to use a CNN (Convolutional Neural Network) as they are the most accurate at image classification; this woudl generate the most precise embeddings possible. We drew inspiration from the VGG face model [from the University of Oxford](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/_) and looked through the documentation to see the different network layers they used to replicate a similar structure. 
2. Generate embeddings for all the images in our dataset using the CNN that we built earlier. These image embeddings will then be processed and passed into an SVM classifier that we build later. 
3. Process the embeddings using PCA into the appropriate format to fit the SVM classifier we build in the next step.
4. Create and fit an SVM classifier. Additionally, we use the Label Encoder class to create a Label Encoder that we can use to output names rather than their numerical encodings. We create the LE in this step as well, and use both it and the SVM classifier to output predictions on our test set.
5. Train the model on a much larger data set than previously used in other celebrity detection examples (dataset creation will be expanded on in the following sections). 
6. Implement the webcam extension using Open CV2 and call the model on frames extracted from the webcam. 

### Data Collection and Preprocessing 
> [MAIN DATASET LINK DOWNLOAD](https://drive.google.com/drive/folders/0B5G8pYUQMNZnLTBVaENWUWdzR0E?resourcekey=0-gRGzioHdCR4zkegs6t1W2Q&usp=sharing)
, [SOURCE](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset)
>
> [DEVELOPMENT DATASET DOWNLOAD](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)


- Main dataset statistics: ~100k images for 1100 celebrities (about 700-800 per celebrity)
- Why this (main) dataset:
We selected this dataset because of its vast array of celebrities including those of different races, gender, and age. It contains differing angles of the celebrities (looking down, to the left, etc). This database was significantly larger than other datasets we found but did come with a potential downside: these images were not hand-screened and came directly from a google search downloaded via a web-crawling python script. Many images also had more than once face. None were cropped to just a celebrity's face. For tesing of the model being trained, we used AT&T's 40-person dataset linked above. To use these images, we created a new script to convert pgm files to png in ```pgm_to_png.py```. A planned source of novelty of this project was the massive source of input.
- Preprocessing: In terms of preprocessing the data, we wrote a script (contained in ```face_crop.py```) which iterated through the images and extracted the face. We also allowed extended this script into a library; one can call our methods from other files in a portable manner. We also implemented checks that threw out data that had more than one face in the image; to select the "right" face would require the model to be trained as samples came in. We also discarded any images that did not have any identifiable faces. Additionally, in face_crop.py, all the images were resized to the same dimensions for uniformity (3 X 224 X 224). Once this processing was complete, the processed images were saved into one new folder to be used in training the model to ensure that the original data would not be overwritten. Images in this folder followed the naming convention "Celebrity Name_Number.png", where number represents 1 for the first image of that celebrity, 2 for the second, and so on. Selecting an output format like this made the model training easier; the model does not have to dive through different directories.

### Pre-existing Components 
> Kaggle Project: https://www.kaggle.com/code/vinayakshanawad/celebrity-face-recognition-vggface-model
>
> Medium Article: https://bencho264.medium.com/face-recognition-with-celebrities-8e2767315fd1

We drew inspiration for building and training our model from the Kaggle project linked above. However, since our dataset was a different size and labelled differently from the one used in that project, we had to modify the layers of the model to fit our data. Additionally, we had to write code to extract the labels differently. 

For preprocessing, we drew inspiration from the Medium article linked, where they describe OpenCV's face cascades, which we also used to extract the faces from our original dataset. 

We also generated a pgm to png conversion script; ```pgm_to_png.py```.

We designed our code to be portable with many options for use (library or script-wise).

### Results  
After writing our project, we trained the model on a much larger dataset than we used for preliminary training while creating the model. This training ran for over 50 hours (on a very sad PC with 16 GB of RAM). Since we didn't want to have end-users have to download the dataset and retrain the model, we used the "pickle" module provided by Python to store the model that we built in train_model.py and load it into celeb_predict.py. 

**FILL IN THE REST OF THIS SECTION LATER, AFTER TRAINING IS COMPLETE**

## Discussion 
### Problems Encountered 
In our initial preprocessing, when an image had more than 1 face, it would save both of those faces as separate images labelled as the celebrity folder the original image was classified under. In order to determine which of those two faces were actually that celebrity, we would have had to manually clean the data or build a pipeline that trained as images were converted; this accuracy would have been low at best. Therefore, we decided that it would be best to discard any images with more than 1 face so that we didn't have incorrectly labelled images in the dataset. For example, if an image of Tom Holland and Zendaya was under the folder labelled "Zendaya", the original implementation would have cropped out both faces and saved the face of Tom Holland as ```Zendaya_1.png``` and the face of Zendaya as ```Zendaya_2.png``` (or vice versa). To avoid having an image of Tom Holland labelled as Zendaya (which would throw off the model), the second implementation would discard that image all together. 

Another problem that we encountered was using the model that we had trained on the image frame captured by the webcam.

From here, we faced issues in transforming the image captured by the webcam into the input format of the model. We used the prior model to generate the embedding but struggled to replicate the normalization and PCA transformations due to the format of this data compared to our training data. This led to inaccuracy in the classification outputted by the SVM at the end. We're currently working on improving the accuracy by adapting those transformations (expanded in next steps).

### Next Steps 
- Improving accuracy by appropriately transforming the embeddings. (PCA transformations currently don't properly apply outside of the train_model file)
- Integrate the database collection script to allow an end-user to customize their celebrity results
- Implementing a popup window that displays an image of the celebrity that the model has classified as most similar to you. 
  - If the webcam extension classified you as "Brittany Spears", a window labelled "Brittany Spears" containing an image of her would 
    pop up   
- Display percent match between the frame captured by the webcam and the celebrity that it was classified as.
- Listing the top 3 matches for celebrity classifications rather than just 1.
- Creating a GUI that allows the user to save and share the prediction outputted by the model.

### Differences in Approach 
Although the training and testing of our model followed a standard approach, we chose to implement the webcam extension to make our project different from others. By choosing to include real time detection, we had to use techniques to save our model to be run on the frames captured by the webcam. This also makes our project more user friendly, as the user does not have to download any datasets or retrain the model. 

