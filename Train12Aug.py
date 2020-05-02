# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:36:24 2019

@author: 19543476
"""


# %% for adding several features to the data
def add_feature(data):
    data["x_y"]=(data.Axis1*data.Axis2) 
    data["y_z"]=(data.Axis2*data.Axis3)
    data["x_z"]=(data.Axis1*data.Axis3)
    data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
    data["Mag_Vec_Feature"] = np.sqrt(data.Multiply) 
    return data 
 
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

def Convolution_Model(time_steps,features):
    model = Sequential()   
    model.add(Conv1D(filters=50, kernel_size=10, activation='relu', input_shape=(time_steps,features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=25, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=7))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model,print(model.input_shape),print(model.output_shape)




# %% 
import pandas as pd
import numpy as np  
import pickle
from sklearn import preprocessing

#sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


#keras packages
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Flatten, core,Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling1D
from keras.utils import np_utils,to_categorical
from keras.models import load_model
#%%scipy packages
from scipy import stats
import scipy as sc

#%%tensorflow
import tensorflow as tf
from tensorflow import keras

#%%other libraries which may be required
import gpxpy
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import mpld3
import datetime
from geopy import distance
from math import sqrt, floor
#import plotly.plotly as py
import plotly.graph_objs as go
import haversine
from math import radians, cos, sin, asin, atan2, degrees
from joblib import Parallel, delayed
import multiprocessing
from functools import reduce
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import h5py    
import pickle
import sys
print(sys.executable)

# %% 

data = pd.read_csv("Train_Final.csv")
pd.value_counts(data['label_name'])

a=data.label_name.unique()
print(a)

act_num=int(input("Enter the number of activities needed"))
values = [int(input('Enter a value: ')) for _ in range(act_num)]



if(act_num == 2): 
   b=((data['label_name']== a[values[0]]) | (data['label_name']== a[values[1]]))
   data=data[b]
else:
     b=((data['label_name']== a[values[0]]) | (data['label_name']== a[values[1]]) | (data['label_name']== a[values[2]]) )
     data=data[b]
     
     
 # %%     
data=add_feature(data)

#%%
LABEL = 'ActivityEncoded'
le = preprocessing.LabelEncoder()
data["label_name"] = le.fit_transform(data["label_name"].values.ravel())
LABELS=list(le.classes_)   

output = open('encoder.pkl', 'wb')
pickle.dump(le,output)
output.close()

#%%

list_x=[]
list_y=[]
list_z=[]
list_Mg=[]
list_x_y=[]
list_y_z=[]
list_x_z=[]
labels=[]


for i in range(len(LABELS)):
    x,y,z,Mg,x_y,y_z,x_z=seperating_features(data,i,60)
    list_x.append(x)
    list_y.append(y)
    list_z.append(z)
    list_Mg.append(Mg)
    list_x_y.append(x_y)
    list_y_z.append(y_z)
    list_x_z.append(x_z)
    labels.append(np.ones((len(x),1))*i)
    
#%%    
'''    
for i in range(len(LABELS)):
    x,y,z,Mg,x_y,y_z,x_z=seperating_features(data,i,60)
    list_x.append(np.column_stack([x,np.ones((len(x),1))*i]))
    list_y.append(np.column_stack([y,np.ones((len(x),1))*i]))
    list_z.append(np.column_stack([z,np.ones((len(x),1))*i]))
    list_Mg.append(np.column_stack([Mg,np.ones((len(x),1))*i]))
    list_x_y.append(np.column_stack([x_z,np.ones((len(x),1))*i]))
    list_y_z.append(np.column_stack([y_z,np.ones((len(x),1))*i]))
    list_x_z.append(np.column_stack([x_z,np.ones((len(x),1))*i]))
    labels.append(np.ones((len(x),1))*i)
'''    
    
    
#%%

list_x=np.array(np.transpose(np.concatenate(list_x, axis=0)))
list_y=np.array(np.transpose(np.concatenate(list_y, axis=0)))
list_z=np.array(np.transpose(np.concatenate(list_z, axis=0)))
list_Mg=np.array(np.transpose(np.concatenate(list_Mg, axis=0)))
list_x_y=np.array(np.transpose(np.concatenate(list_x_y, axis=0)))
list_y_z=np.array(np.transpose(np.concatenate(list_y_z, axis=0)))
list_x_z=np.array(np.transpose(np.concatenate(list_x_z, axis=0)))    


#%%

final2=[list_x,list_y,list_z,list_Mg,list_x_y,list_y_z,list_x_z]
final2=np.array(final2) 
final2=np.transpose(final2)


#%%
labels= np.array((np.concatenate(labels, axis=0)))
labels = np.asarray(labels)        
labels = to_categorical(labels) 




#%%
model,inpu,output=Convolution_Model(60,7)


#%%        
epochs,batch_size=30,50
model.fit(final2, labels, epochs=epochs, batch_size=batch_size)

#%%            

model.save('Final_Model.h5')  

#%%    
        
    










  