from data_loader import Data_loader
from preprocessing import Preprocessing
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
class My_Model(Preprocessing):
    
    def custom_cnn(cnn,input_shape):
        model=Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape= input_shape,padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2),padding='same'))

        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2),padding='same'))

        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2),padding='same'))

        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2),padding='same'))


        model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2),padding='same'))

        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(2,activation='softmax'))

        return model

    def model_compile(self,input_shape):
        model=self.custom_cnn(input_shape)
        model.compile(loss="categorical_crossentropy",optimizer='adam',metrics='accuracy')
        return model

    def fit_model(self,test_split,input_shape,epochs,batch_size):
        x_train,x_test,y_train,y_test= super().data_pre_processing(test_split)
        classifier=self.model_compile(input_shape)
        classifier.summary()
        history=classifier.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)
        return history