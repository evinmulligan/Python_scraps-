# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:30:13 2023

@author: Evin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre

#       Loads the data
data = np.loadtxt(r'C:\Users\evinm\Downloads\ProjectDigits_materials(1)\ProjectDigits_materials\mfeat-pix.txt')
datadf = pd.DataFrame(data)

#       Plots 10 digits from each class
fig, axs = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []})
for i in range(10):
    for j in range(10):
        picmat = data[200*i + j].reshape(16,15, order = 'A')
        axs[i,j].pcolormesh(np.flip(picmat, 0))
        
#       Script to determine whether a string is a subsequence of another
#       This will be used for the radiality conditions, e.g detect whether
#       the pixels go from black to white to black again
def isSubsequence(x, y):
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)

#       The strings we will look for, 6 is black, 0 is white, so 606 represents
#       a transition from black to white to black again, as in the number 0
lines2 = '606'
lines3 = '60606'

#       Here we create vectors for the radiality conditions, and one for the brightness
Radiality = np.zeros((2000,5))
Brightness = np.zeros(2000)
for i in range(2000):
    # write the image as an array. Note it must be flipped to fix the orientation
   picmat = np.flip(data[i].reshape(16,15, order = 'A'),0)
   
   # image brightness is the mean of the pixels brightness
   Brightness[i] = np.mean(data[i,:])
   
   # convert the float to integers, then to a string, and take the central
   # row and column, that is, the horizontal and vertical lines through the centre
   # of the image. We can then test for transitions from black to white.
   picmatint = np.array(picmat, dtype = int)
   horz = ''.join(str(x) for x in picmatint[8,:]) 
   vert = ''.join(str(x) for x in picmatint[:,7]) 

    # if the image goes from b-w-b horizontally
   if isSubsequence(lines2, horz) == True:
       Radiality[i,0] = 1
    # if it goes b-w-b vertically   
   if isSubsequence(lines2, vert) == True:
       Radiality[i,1] = 1
    # b-w-b-w-b vertically
   if isSubsequence(lines3, vert) == True:
       Radiality[i,2] = 1
    # this createa vectors for the above horizontal condition combined with either
    # vertical condition
   Radiality[i,3] = Radiality[i,0]*Radiality[i,1]
   Radiality[i,4] = Radiality[i,0]*Radiality[i,2]

#   save the custom radiality features and the brightness feature in data frames, then label the columns
raddf = pd.DataFrame(Radiality, columns = ['rad0', 'rad1', 'rad2', 'rad3', 'rad4', ])       
brightdf = pd.DataFrame(Brightness, columns = ['brightness'])

#   This creates "prototype" vectors, that are averages of the examples of each digit
#   and measures the match, that is the dot product of each image with each prototype
#   We expect high values for images from the same class.
prototype = np.zeros((10,240))
ProtMatch = np.zeros((2000,10))
for i in range(10):
    prototype[i,:] = np.mean(data[0:199+i*200,:], axis = 0)
    ProtMatch[:,i] = np.dot(prototype[i,:], np.transpose(data))

#   saves the results in a data frame, and find which prototype best matches each image
protdf = pd.DataFrame(ProtMatch)     
colindex = protdf.idxmax(axis = 1)

#   Add our custom features to the new dataframe.
df = pd.concat([datadf, raddf, brightdf, protdf,], axis=1, join='inner')




       
