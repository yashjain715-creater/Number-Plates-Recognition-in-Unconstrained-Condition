'''
This will be main file which the co-ordinaters of the event will be using to test your
code. This file contains two functions:

1. predict: You will be given an rgb image which you will use to predict the output 
which will be a string. For the prediction you can use/import code,models from other files or
libraries. More detailes given above the function defination.

2. test: This will be used by the co-ordinators to test your code by giving sample 
inputs to the 'predict' function mentioned above. A sample test function is given for your
reference but it is subject to minor changes during the evaluation. However, note that
there won't be any changes in the input format given to the predict function.

Make sure all the necessary functions etc. you import are done from the same directory. And in 
the final submission make sure you provide them also along with this script.
'''

# Essential libraries and your model can be imported here
import os
import cv2
import numpy as np 
from tensorflow.keras.models import load_model
import segmentation as segment
from keras.models import model_from_json
import local_utils



image_size = 28

'''
function: predict
input: image - A numpy array which is an rgb image
output: answer - A list which is the full word

Suggestion: Try to make your code understandable by including comments and organizing it. For 
this we encourgae you to write essential function in other files and import them here so that 
the final code is neat and not too big. Make sure you use the same input format and return 
same output format.
'''
def predict(image):
    '''
    Write your code for prediction here.
    '''
    print("#############################")
    plates = []
    answer = []
    wpod = False
    scale_factor = 6
    print(image.shape)
    
    #This part of code need to be modeifired a bit for using the WPOD part(i.e. detecting the plate from car image) 

    if(image.shape[0]*1.5<image.shape[1]):
        plates.append(image)
        scale_factor = 6
        print("LP***************************")
    else:
        float_plates, coord = local_utils.get_plate(image)
        if(float_plates is not None):
            print("WPOD***************************")
            wpod = True
            scale_factor = 3
            for img in float_plates:
                temp = (img*255)
                temp=np.uint8(temp) 
                plates.append(temp)  
        else:
            print("LP***************************")
            plates.append(image)
    
    for plate in plates:
        # path = "Chevrolet-Captiva-524767e.jpg_0348_0049_0207_0136_0054.png"
        # plate = cv2.imread(path)
        plate = cv2.resize(plate,(scale_factor*plate.shape[1],scale_factor*plate.shape[0]))
        # cv2_imshow(plate)

        hull = segment.CreateHull(plate)

        mask = segment.createMask(plate.shape[0:2], hull)
        # cv2_imshow(mask)

        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print((contours))
        # cv2.drawContours(plate, contours, -1, (0,0,255), 1)
        # cv2_imshow(plate)

        temp = contours[0]
        hull2 = cv2.convexHull(temp, False)
        perimeter = cv2.arcLength(hull2,True) #it gives the perimeter of each shape(true is for closed shape)
        approxCnt = cv2.approxPolyDP(hull2,0.03*perimeter,True) #this will give coridinates of all the corner points
        # cv2.drawContours(hull2,[approxCnt],0,color =(0,0,255),thickness = 3)
        # cv2.drawContours(plate, [approxCnt], 0, (0,0,255), 3)
        # segment.cv2_imshow(plate)
        No_of_points = len(approxCnt)
        if No_of_points == 4 and not wpod:
            wrap = segment.four_point_transform(plate, np.array([approxCnt[0][0],approxCnt[1][0],approxCnt[2][0],approxCnt[3][0]]))

        else:
            wrap = plate
        wrap = cv2.fastNlMeansDenoisingColored(wrap,None,10,10,7,21)
        wrap = cv2.cvtColor(wrap,cv2.COLOR_BGR2GRAY)
        # segment.cv2_imshow(wrap) 
        Iopen = segment.ClearThresold(wrap)
        # segment.cv2_imshow(Iopen)
        segmented = segment.Get_Segmented(Iopen)

        json_file = open('MobileNets_character_recognition.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("License_character_recognition_weight.h5")    
        characters = ["0",   "1",   "2",   "3",   "4",   "5",   "6",   "7",   "8",   "9",   "A",   "B",   "C",   "D",   "E",   "F",   "G",   "H",   "I",   "J",   "K",   "L",   "M",   "N",   "O",   "P",   "Q",   "R",   "S",   "T",   "U",   "V",   "W",   "X",   "Y",   "Z"]  
        ANSWER = []
        z= 1
        for image in segmented:
            # cv2.imshow("image"+str(z),image)#this imshow dhows the final segmented characters
            z+=1
            ANSWER.append(characters[segment.predict_from_model(image,model)])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(ANSWER)
        ANSWER = segment.refine_answer(ANSWER)# modifying answers for more accuracy using indian number plates rules
        print(ANSWER)
        answer.append(ANSWER)
    return answer


'''
function: test
input: None
output: None

This is a sample test function which the co-ordinaors will use to test your code. This is
subject to change but the imput to predict function and the output expected from the predict
function will not change. 
You can use this to test your code before submission: Some details are given below:
image_paths : A list that will store the paths of all the images that will be tested.
correct_answers: A list that holds the correct answers
score : holds the total score. Keep in mind that scoring is subject to change during testing.

You can play with these variables and test before final submission.
'''
def test():
    '''
    We will be using a similar template to test your code
    '''
    # image_paths = ["./images/Toyota-Etios-Liva-521420d.jpg_0000_0562_0341_0086_0054.png"]
    # image_paths = ["./ps2/test_multipleCar/p5.png"]
    image_paths = ["./ps2/test_pics/t4.png"]

    score = 0
    multiplication_factor=2 #depends on character set size

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected



if __name__ == "__main__":
    test()



