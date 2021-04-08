import sys
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

class CnnImageClassifier:
    
    # define cnn model
    @staticmethod
    def define_CnnModel_Architecture():
        # keeping dimensions of image 256 by 256 for clear presentation of bounding box
        img_width, img_height = 256, 256
        input_shape=''
        #decide which type of input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        #add 2d pooling layer with max pooling
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        #add 2d pooling layer with max pooling
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        #add 2d pooling layer with max pooling
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add 2d pooling layer with max pooling
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add a convolution layer with 64 filters and filter size (3,3)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        #add 2d pooling layer with max pooling
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
          #add aflatten layer
        model.add(Flatten())
        #add dense layer
        model.add(Dense(units=4096,activation="relu"))
        #add dense layer
        model.add(Dense(units=4096,activation="relu"))
        #add dense layer
        model.add(Dense(units=2, activation="softmax"))
        opt = Adam(lr=0.001)
        # compiling final model
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # run this method for evaluating the model
    @staticmethod
    def evaluate_model(model, generator):
        #generate accuracy fo model from validation split of data set
        score = model.evaluate_generator(generator=generator,steps=generator.samples//nBatches)   
        print("%s: Model evaluated:"
              "\n\t\t\t\t\t\t Loss: %.3f"
              "\n\t\t\t\t\t\t Accuracy: %.3f" %
              (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),score[0], score[1]))
    
    # run this method to train the CNN model using training data set
    @staticmethod
    def trainCnnImageClassifier():
        #get the current working directory
        current_working_directory = os.getcwd()
        #current_working_directory ="/Users/vikas/Documents/model/"
        #obtain the model file path to save model after training
        modelFilePath = current_working_directory + "cnnmodel.h5"
        # dimensions of our images.
        img_width, img_height = 256, 256
        #path of training data set
        train_data_dir = '/Users/vikas/Documents/data/train'
        #path of test data set
        validation_data_dir = '/Users/vikas/Documents/data/validation'
        #number of images in training data set for all 5 classes
        nb_train_samples = 6500
        #number of images in validation dataset for all 5 classses
        nb_validation_samples = 1000
        #number of epochs to run the training
        epochs = 50
        batch_size = 32
        # load defined model architecture
        model = CnnImageClassifier.define_CnnModel_Architecture()
        
        #add early stopping only save models if accuracy has improved after completion of an epoch
        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
        #create checkpoints to save model after completion of every epoch
        checkpoint = ModelCheckpoint(modelFilePath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

        
        # obtain ImageDataGenerator object for traininig
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # obtain ImageDataGenerator object for testing
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        
        #create Data generator for training the model
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        #create Data generator for testing and validating the model
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        #train and fit the model
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks = [early_stop,checkpoint])

        #saving trained model file which could be loaded to make predictions, saved in the current working directory of the program
        print("Saved model to disk in current working directory")
        # obtain model accuracy
        CnnImageClassifier.evaluate_model(model,validation_generator)
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
     # run the test harness for evaluating a model
    @staticmethod
    def loadTrainedCnnModel():
        # load the model
        #
        #
        #
        #
        #
        #
        #
        model = VGG16(weights='imagenet')
        print("Trained model has been loaded")
        # summarize model.
        model.summary()
        return model
        
        
#only invoke the train method in case of trainnig the model
#CnnImageClassifier.trainCnnImageClassifier()
#CnnImageClassifier.loadTrainedCnnModel()

