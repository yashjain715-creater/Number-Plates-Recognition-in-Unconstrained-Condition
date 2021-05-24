# Number Plates Recognition in Unconstrained Condition

This is our work on Number Plates Recognition in Unconstrained Condition presented at Mosaic'21 round 2. The problem statement was "RAPID AI vision". By unconstrained what we mean is:

![sample](https://github.com/Bellicose-YB/Number-Plates-Recognition-in-Unconstrained-Condition/blob/main/Images/Capture.PNG)

## Major Subdivisions in our Approach:
* Number Plate Detection from image of car
* Perspective Transform
* segmentation.
* Prediction
![sample](https://github.com/Bellicose-YB/Number-Plates-Recognition-in-Unconstrained-Condition/blob/main/Images/profile2.jpg)


## WORKFLOW:

* ![Markdown Logo](https://github.com/Bellicose-YB/Number-Plates-Recognition-in-Unconstrained-Condition/blob/main/Images/Professional_profile.jpg)

## Prediction of Segmented Letters:

* We used a pretrained model whose reference we got from medium website.
They used mobilenet architecture with pre trained weight on imagenet dataset.
Link:https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-3-recognize-be2eca1a9f12




## To run on your local system:
* Clone the repository
* Carefully check on which image you want to run as well as it is in images folder or not.
* Finally run main.py(driver code).

## FEATURES:
* Performs good even on skew images, noisy and distorted images.
* we have also made use of WPOD net network which helps number plate detection in images and videos of car.

* Car number plate detction from video 
Link:https://drive.google.com/drive/folders/134Yp0uuv01s7-lwQjCtnKfHe_nbWYUJM?usp=sharing 
