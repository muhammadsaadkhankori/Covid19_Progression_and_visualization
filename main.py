from data_loader import Data_loader
from preprocessing import Preprocessing
from model import My_Model
path=r'E:\Study materials\Torch\Dataset\*'
img_size=(224,224)
data=Data_loader(path,img_size)
data=Preprocessing(path,img_size)
#x_train,x_test,y_train,y_test = data.data_pre_processing(0.1)
data=My_Model(path,img_size)
input_shape=(224,224,3)
data.model_compile(input_shape)
epochs,batch_size=2,32
data.fit_model(0.1,input_shape,epochs,batch_size)