import numpy as np
from data_loader import Data_loader
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Preprocessing(Data_loader):
    def __init__(self,path,img_size):
        super().__init__(path,img_size)

    def data_pre_processing(self, test_split):
        images,labels = super().data_loader()
        images=np.array(images) #Converting images  into numpy array
        labels=np.array(labels) #Converting labels  into numpy array
            
        label_encoder=LabelEncoder()
        labels=label_encoder.fit_transform(labels)
        images=images / 255.
        print("Images",images.shape)
        print("Labels",labels.shape)
        x_train,x_test,y_train,y_test = train_test_split (images,labels,test_size=test_split,random_state=True)
        y_train = to_categorical(y_train)
        return x_train,x_test,y_train,y_test
    