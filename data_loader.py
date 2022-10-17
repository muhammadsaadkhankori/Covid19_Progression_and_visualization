import os,cv2,glob,random
import matplotlib.pyplot as pyplot

class Data_loader():

    def __init__(self,path,img_size):
        self.path = path
        self.img_size=img_size

    def data_loader(self):

        data_paths = os.path.join(self.path,'*g')
        imagePaths = glob.glob(data_paths)
        print("Total images",len(imagePaths)) #Images in dataset

        images=[]
        labels=[]

        for imgpath in imagePaths:
            img=cv2.imread(imgpath)
            img=cv2.resize(img,self.img_size)

            images.append(img)

            label=imgpath.split(os.path.sep)[-2].split("_")
            labels.append(label)

        return images,labels

    

