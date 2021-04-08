from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib import rcParams

   
class GradCamClass:

    
    def __init__(self):
        print("in init")

    # load and prepare the image
 
    def load_image(self,filename):
        # load the image
        image = load_img(filename, target_size=(224, 224))
        # convert to array
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

      # predict image label and draw boudning box
    
    def sayHello(self):
        print("hello world")
    
    # predict image label and draw boudning box
    def predictAndLocalizeObject(self,model,imagePath):
        #load image for prediction
        imageObj = self.load_image(imagePath)
        #predict object class in the image with the CNN trained model
        predictedOutcome = model.predict(imageObj)
        #print('Predicted:', decode_predictions(predictedOutcome, top=3)[0])
        # convert the probabilities to class labels
        label = decode_predictions(predictedOutcome)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0][1]
        print("label",label)
        # print the classification
        outcomeArgMax=np.argmax(predictedOutcome[0])
        #obtain prediction map of the predicted class
        predictedClassDefaultMap= model.output[:, outcomeArgMax]
        #extract last convolution layer of the neural network
        last_conv_layer = model.get_layer('block5_conv3')
        #obtain gradient predicted class map and output of the last concolution layer
        gradientObj = K.gradients(predictedClassDefaultMap, last_conv_layer.output)[0] 
        #obtain pool gradients
        pooled_grads = K.mean(gradientObj, axis=(0, 1, 2))
        #print(last_conv_layer.output[0])
        iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([imageObj])
        #print("length of arrays : ",pooled_grads_value.shape,conv_layer_output_value.shape)
        
        #iterate over all the values of image vector
        for i in range(256):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        #generate heatmap from the output of of the last convolution layer
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #plot the heat map
        #plt.matshow(heatmap)
        
        #read the original image with CV
        originalImageObj = cv2.imread(imagePath)
        current_working_directory = os.getcwd()
        originalImagePath = current_working_directory +"/originalImage.jpg"
        cv2.imwrite(originalImagePath,originalImageObj)
        #resize the heatmap according to the original image size
        heatmap = cv2.resize(heatmap, (originalImageObj.shape[1], originalImageObj.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #get the current working directory
        current_working_directory = os.getcwd()
        temp_Image_Heatmap_Path = current_working_directory +"/image_Heatmap_Temp.jpg"
        #save the heat map in temporary location
        cv2.imwrite(temp_Image_Heatmap_Path, heatmap* 0.4)
        # Grayscale then Otsu's threshold
        # read the saved heatmap
        heatMapImageObj = cv2.imread(temp_Image_Heatmap_Path)
        #save the original image in a temp location for processing
        temp_Image_Path=current_working_directory +"/originalImageTemp.jpg"
        cv2.imwrite(temp_Image_Path,originalImageObj)
        #obgain object of originalimage saved in temporary location
        tempImageObj = cv2.imread(temp_Image_Path)
        # Grayscale then Otsu's threshold
        gray = cv2.cvtColor(heatMapImageObj, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow('thresh', thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        i=1;
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if i==1:
                 height, width, channels = tempImageObj.shape
                 #print(y,height)
                 if(y>7):
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    #print(y+5,height-3)
                    cv2.putText(tempImageObj, label, (x,y-3), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    i=0
                 else :
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(tempImageObj, label, (x,y+h+21), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    i=0
            cv2.rectangle(tempImageObj, (x, y), (x + w, y + h), (36,255,12), 2)
        
        #save the bound boxed original Image copy
        cv2.imwrite(temp_Image_Path,tempImageObj)
        
         
        rcParams['figure.figsize'] = 20,8
        image1path = os.getcwd() +"/originalImage.jpg"
        image2path = os.getcwd() +"/image_Heatmap_Temp.jpg"
        image3path =  os.getcwd() +"/originalImageTemp.jpg"
        
        # read images
        img_A = mpimg.imread(image1path)
        img_B = mpimg.imread(image2path)
        img_C = mpimg.imread(image3path)
        # display Output images
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(img_A);
        ax[0].set_title('Input-Image')
        ax[1].imshow(img_B);
        ax[1].set_title('Localized HeatMap')
        ax[2].imshow(img_C);
        ax[2].set_title('Classifier-Output')
        plt.show()
        cv2.destroyAllWindows() 
        print("Image processed")
            
    