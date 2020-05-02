# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:56:12 2019

@author: 19176368
"""

#%%Functions
####Data segmenting

def seperating_features(data,activity,roll_size):  
    #rolling each of the columns in the data and loading them as different arays
    x=roll_ts(data[data["label_name"]==activity].Axis1,roll_size)
    y=roll_ts(data[data["label_name"]==activity].Axis2,roll_size)
    z=roll_ts(data[data["label_name"]==activity].Axis3,roll_size)
    Mg=roll_ts(data[data["label_name"]==activity].Mag_Vec_Feature,roll_size)
    x_y=roll_ts(data[data["label_name"]==activity].x_y,roll_size)
    y_z=roll_ts(data[data["label_name"]==activity].y_z,roll_size)
    x_z=roll_ts(data[data["label_name"]==activity].x_z,roll_size)
    return x, y, z, Mg, x_y, y_z, x_z
    

#for rolling different columns of the data 
def roll_ts(series,window):
    out=np.tile(series,(len(series)-window+1,1))
    index=np.empty([len(series)-window+1,window],dtype=int)
    for i in range(0,len(series)-window+1):
        index[i]=np.arange(i,i+window)
    return out[0,index]    


#for plotting the confusion matrixS    
def show_confusion_matrix(validations, predictions ,LABELS):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()    
 
#for concatinating the output obtained from seperate_feature() function     
def concat(Idle,Walk,grazing,licking,rumination):
    concat=np.concatenate((Idle,Walk,grazing,licking,rumination),axis=0)
    concat=np.transpose(concat)
    return concat
#for creating the model 
def Convolution_Model(time_steps,features):
    model = Sequential()   
    model.add(Conv1D(filters=50, kernel_size=10, activation='relu', input_shape=(time_steps,features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=25, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=7))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model,print(model.input_shape),print(model.output_shape)

#for adding several features to the data
def add_feature(data):
    data["x_y"]=(data.Axis1*data.Axis2) 
    data["y_z"]=(data.Axis2*data.Axis3)
    data["x_z"]=(data.Axis1*data.Axis3)
    data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
    data["Mag_Vec_Feature"] = np.sqrt(data.Multiply) 
    return data 
#%% Libraries
#pandas and numpy

import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
#keras packages
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical




#%%Reading the whole data
data = pd.read_csv("Train_Final.csv")

#%%Adding more features to the data
data=add_feature(data)

#%%Encoding the class labels
# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
data["label_name"] = le.fit_transform(data["label_name"].values.ravel())
LABELS=list(le.classes_)   

#%%Saving the encoder for encoding the test data similarly 
output = open('encoder.pkl', 'wb')
pickle.dump(le, output)
output.close()
#data["x_y"]=(data.Axis1*data.Axis2)
#data["y_z"]=(data.Axis2*data.Axis3)
#data["x_z"]=(data.Axis1*data.Axis3)
#data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
#data["Mag_Vec_Feature"] = np.sqrt(data.Multiply)
# Define column name of the label vector
#LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
#le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
#data["label_name"] = le.fit_transform(data["label_name"].values.ravel())
#output = open('encoder.pkl', 'wb')
#pickle.dump(le, output)
#output.close()
#LABELS=list(le.classes_)    
#print(list(le.classes_))   

#%%Seperating all features for each activity.
#the value for the activity variable in seperation_feature() function should be given based upon the encoding output optained i.e LABELS 
#seperating_features(data,activity,roll_size)
#Idle
Idle_x,Idle_y,Idle_z,Idle_Mg,Idle_x_y,Idle_y_z,Idle_x_z=seperating_features(data,0,30)
#%%Walk
Walk_x,Walk_y,Walk_z,Walk_Mg,Walk_x_y,Walk_y_z,Walk_x_z=seperating_features(data,4,30)
#%%grazing
grazing_x,grazing_y,grazing_z,grazing_Mg,grazing_x_y,grazing_y_z,grazing_x_z=seperating_features(data,1,30)
#%%grazing
licking_x,licking_y,licking_z,licking_Mg,licking_x_y,licking_y_z,licking_x_z=seperating_features(data,2,30)
#%%rumination
rumination_x,rumination_y,rumination_z,rumination_Mg,rumination_x_y,rumination_y_z,rumination_x_z=seperating_features(data,3,30)

#%%concating the results obtained from seperating_features() functions
#concat(Idle,Walk,grazing,licking,rumination), 
x1=concat(Idle_x,Walk_x,grazing_x,licking_x,rumination_x)
x2=concat(Idle_y,Walk_y,grazing_y,licking_y,rumination_y)
x3=concat(Idle_z,Walk_z,grazing_z,licking_z,rumination_z)
x4=concat(Idle_Mg,Walk_Mg,grazing_Mg,licking_Mg,rumination_Mg)
x5=concat(Idle_x_y,Walk_x_y,grazing_x_y,licking_x_y,rumination_x_y)
x6=concat(Idle_y_z,Walk_y_z,grazing_y_z,licking_y_z,rumination_y_z)
x7=concat(Idle_x_z,Walk_x_z,grazing_x_z,licking_x_z,rumination_x_z)

#%%creating reshaping the training dataset 
final=[x1,x2,x3,x4,x5,x6,x7]
final=np.array(final)
final=np.transpose(final)

#%%processing the dependent variable

I,W,R,L,G=pd.DataFrame(Idle_x),pd.DataFrame(Walk_x),pd.DataFrame(rumination_x),pd.DataFrame(licking_x),pd.DataFrame(grazing_x)


I["label_name"]=0
W["label_name"]=4
G["label_name"]=1
L["label_name"]=2
R["label_name"]=3

y1,y2,y3,y4,y5=I["label_name"],W["label_name"],G["label_name"],L["label_name"],R["label_name"]


label=pd.concat([y1,y2,y3,y4,y5],axis=0)

#one hot encoding 
label = np.asarray(label)        
label = to_categorical(label)  

del I,W,R,L,G

#%%Traing and Testing split
#from sklearn.model_selection import train_test_split
#X_Train,X_Test,Y_Train,Y_Test= train_test_split(final,label,test_size=0.2,random_state=0)
#%%creating the 1D convolutional neural network model
#Convolution_Model(time_steps,features)
model,inpu,output=Convolution_Model(30,7)

#%%Fitting the model with the training data    
epochs,batch_size=30,50
model.fit(final, label, epochs=epochs, batch_size=batch_size)
  
#%%validating the model with test dataset 
#y_pred_test = model.predict(final)
    
    
#max_y_pred_test = np.argmax(y_pred_test, axis=1)
#max_y_test = np.argmax(label, axis=1)
#show_confusion_matrix(max_y_test, max_y_pred_test,LABELS)
#print(classification_report(max_y_test, max_y_pred_test))

#%% Save Model
model.save('Final_Model1.h5')  
    