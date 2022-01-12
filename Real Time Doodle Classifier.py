#importing all the necessary Libraries

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader,random_split
import math
from torch import optim  
from torch import nn 
import torch.nn.functional as F 
import cv2

#Creating a Class

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=20):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,8,3,stride=1,padding=1)    #Convulation(Stride=1,Padding=1)
        self.pool = nn.MaxPool2d(2,2)                                 #Pooling(Which reduces the Dimension of image by half)
        self.conv2 = nn.Conv2d(8,16,3,stride=1,padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):                            #Forward Function
        x = F.relu(self.conv1(x))                      #Convulation on Image
        x = self.pool(x)                               #Pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)                   #Reshaping The image into a Column Vector/Matrix
        x = self.fc1(x)
        return x

# Defining Classes Of Different Objects

classes=('airplane','ant','banana','baseball','bird','bucket','butterfly','cat','coffee cup','dolphin','donut','duck','fish','leaf','mountain','pencil','smiley face','snake','umbrella','wine bottle')


#Defining The no of Channels In Input Image and No of classes

in_channels=1
num_classes=20

#Loading the File
FILE=".vscode\model.pth"    #Enter the adress of Your File
loaded_model = CNN(in_channels, num_classes)
loaded_model.load_state_dict(torch.load(FILE)) 
loaded_model.eval()


#Creating A Drawing Pad using Opencv Library

drawing = False 
pt1_x , pt1_y = None , None
l=0
# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing,l

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y
        
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=20)
            pt1_x,pt1_y=x,y
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=20)        
        l=27

img = np.zeros((500,500,3), np.uint8)   #Defining The Size of Drawing Pad
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    p=cv2.waitKey(1) & 0xFF
   
    if p==ord('z'):           #Press 'z' To end the program
        break
    if p==ord('a'):           #Press 'a' To draw a new Object
        img = np.zeros((500,500,3), np.uint8)
    if l==27:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_img=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(input_img)
        x=x.reshape(1,1,28,28)

        res=loaded_model(x.float())
        i=torch.argmax(res)

        print("I Guess It is a ",classes[i.item()])
        l=0
    
    
cv2.destroyAllWindows()