# impot pakages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data creat
import os
path = "..\\data refining\\"
file= os.listdir(path)[:84]
print(file)
classes={'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':8,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,
         '15':14,'16':15,'17':16,'18':17,'19':18,'20':19,'21':20,'22':21,'23':22,'24':23,'25':24,'26':25,'27':26,'28':27,
         '29':28,'30':29,'31':30,'32':31,'33':32,'34':33,'35':34,'36':35,'37':36,'38':37,'39':38,'40':39,'41':40,'42':41,
         '43':42,'44':43,'45':44,'46':45,'47':46,'48':47,'49':48,'50':49,'51':50,'52':51,'53':52,'54':53,'55':54,'56':55,
         '57':56,'58':57,'59':58,'60':59,'61':60,'62':61,'63':62,'64':63,'65':64,'66':65,'67':66,'68':67,'69':68,'70':69,
         '71':70,'72':71,'73':72,'74':73,'75':74,'76':75,'77':76,'78':77,'79':78,'80':79,'81':80,'82':81,'83':82,'84':83}

import cv2 # to convert into array or resize the img.
x=[]
y=[]
for c in classes:
     p=path+c
     for img in os.listdir(p):
         img= cv2.imread(p+"/"+img,0)
         x.append(img)
         y.append(classes[c])
pd.Series(y).value_counts()
print(x[0].shape)
x = np.array(x)
y= np.array(y)
print(len(x))
plt.imshow(x[100],cmap="gray") # after cheak for the all imp.
#print(y[0])
#print(x.shape)

#prepare the data
x_new= x.reshape(len(x),-1)

# split the data for train and test
xtr,xte,ytr,yte= train_test_split(x_new,y,test_size=.30,random_state=30)
mean = np.mean(xtr, axis=0)
std = np.std(xtr, axis=0)

xtr = (xtr - mean) / std
xte = (xte - mean) / std
# Logistic Regression Model
model = LogisticRegression()
model.fit(xtr, ytr)

# Prediction and Evaluation
y_pred = model.predict(xte)
accuracy = accuracy_score(yte, y_pred)

print("Accuracy:", accuracy)
