# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:51:02 2019

@author: 19543476
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:13:02 2019

@author: 19176368
"""

#for adding several features to the data
def add_feature(data):
    data["x_y"]=(data.Axis1*data.Axis2) 
    data["y_z"]=(data.Axis2*data.Axis3)
    data["x_z"]=(data.Axis1*data.Axis3)
    data["Multiply"]=((data.Axis1*data.Axis1)+(data.Axis2*data.Axis2)+(data.Axis3*data.Axis3))
    data["Mag_Vec_Feature"] = np.sqrt(data.Multiply) 
    return data 


#for plotting the confusion matrixS    
def show_confusion_matrix(validations, predictions ,LABELS):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
        
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()   

#for preprocessing the test data    
def test_preprocess(time_steps,step):
    # Number of steps to advance in each iteration (for me, it should always 
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(data) - time_steps, step):
        xs = data['Axis1'].values[i: i + time_steps]
        ys = data['Axis2'].values[i: i + time_steps]
        zs = data['Axis3'].values[i: i + time_steps]
        x_y = data['x_y'].values[i: i + time_steps]
        y_z = data['y_z'].values[i: i + time_steps]
        x_z = data['x_z'].values[i: i + time_steps]
        Mag= data['Mag_Vec_Feature'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = sc.stats.mode(data["label_name"][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs, Mag, x_y, y_z, x_z])
        labels.append(label)
    #one hot encoding    
    labels = np.asarray(labels)
    labels = to_categorical(labels)    
    return segments,labels
        
    

    
     
#%%
import numpy as np   
import pandas as pd
import gpxpy
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import mpld3
import datetime
from geopy import distance
from math import sqrt, floor
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
import scipy.io  

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
import pickle    
from scipy import stats
import scipy as sc

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

#%%Reading the Raw Test data

data = pd.read_csv("Test_gwr_Final.csv")
#%%Reading the label encoder file written in Train file
pkl_file = open('encoder.pkl', 'rb')
le_new = pickle.load(pkl_file) 

#%%Adding the fetures
data=add_feature(data)

#%%Encoding the class labels
# Add a new column to the existing DataFrame with the encoded values
data["label_name"] = le_new.transform(data["label_name"].values.ravel())
print(list(le_new.classes_)) 

#%%preprocessing the test data
#test_preprocess(time_steps,step),advisable to go with value 30 for both step and time_steps
test,labels=test_preprocess(60,60)

#%%reshaping the test data
test=np.array(test)

test=np.transpose(test,(0,2,1))
#%%testing the test data with the model
new_model =  load_model('Final_Model.h5') 
y_pred_test = new_model.predict(test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(labels, axis=1)
show_confusion_matrix(max_y_test, max_y_pred_test,labels)
print(classification_report(max_y_test, max_y_pred_test))
#%%
