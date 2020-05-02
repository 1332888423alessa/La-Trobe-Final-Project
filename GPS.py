import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['PROJ_LIB'] ='/Users/mac/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

fileName = pd.read_csv("GPS.csv")

plt.figure(figsize=(10,10))
m = Basemap()
m.drawcoastlines()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='white',lake_color='white')
parallels = np.arange(-90., 90., 10.) #latitude range​
m.drawparallels(parallels,labels=[False, True, True, False])
meridians = np.arange(-180., 180., 20.) #longitude range​
m.drawmeridians(meridians,labels=[True, False, False, True])
lat = np.array(fileName["latitude"] [0:6]) #total 6 data​
lon = np.array(fileName["longitude"] [0:6])
x,y = m(lon, lat)
m.scatter(x, y, s=100, color = 'r')
plt.title('Location Map')
plt.show()