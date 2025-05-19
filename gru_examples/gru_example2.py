#Change the values to run the experiment

experiment = 1 #1-> BANKEX , 2 ->Activities
f = 20 #Size of future window in days. Change f for the number of steps-ahead. 1 or 20.
thepath = 'sample_data' #Change the path to where you want to save the models

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from numpy import genfromtxt
#from pandas_datareader import data as pdr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import mannwhitneyu


def GRU_Model(output_window):
  model = Sequential()
  model.add(GRU(128, return_sequences=False, input_shape=(X.shape[1],1)))
  model.add(Dense(output_window))
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.summary()
  return model


#Based on: Computer Science: Stock Price Prediction Using Python & Machine Learning. https://www.youtube.com/watch?v=QIUxPv5PJOY (2019), accessed 25 January,
2022
def data_preparation(w, scaled_data, N, f):
  X=[]
  window = w + f
  Q = len(scaled_data)
  for i in range(Q-window+1):
    X.append(scaled_data[i:i+window, 0])

  X = np.array(X)
  X = np.reshape(X, (X.shape[0],X.shape[1],1))

  trainX, trainY = X[0:N,0:w], X[0:N,w:w+f]
  testX, testY = X[N:Q-w,0:w], X[N:Q-w,w:w+f]
  X = trainX 
  return trainX, trainY, testX, testY, X

#Repeats the last known value f times
def baselinef(U,f):
  last = U.shape[0]
  yhat = np.zeros((last, f))
  for j in range(0,last):
    yhat[j,0:f] = np.repeat(U[j,U.shape[1]-1], f)
  return yhat

#Directional accuracy
#From https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def mda(actual: np.ndarray, predicted: np.ndarray):
  return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

#Normalize data between 0 and 1
def scaleit(DATAX):
  mima = np.zeros((DATAX.shape[0], 2)) #To save min and max values
  for i in range(DATAX.shape[0]):
    mima[i,0],mima[i,1] = DATAX[i,:].min(), DATAX[i,:].max()
    DATAX[i,:] = (DATAX[i,:]-DATAX[i,:].min())/(DATAX[i,:].max()-DATAX[i,:].min())
  return DATAX, mima
#Inverse normalization
#Rescale to original values
def rescaleit(y,mima,i):
  yt = (y*(mima[i,1]-mima[i,0]))+mima[i,0]
  return yt

#Based on https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
#This code is to plot series of different colors
def plot_series(X):
  x = np.arange(10)
  ys = [i+x+(i*x)**2 for i in range(10)]
  colors = cm.rainbow(np.linspace(0, 1, len(ys)))
  for i in range(10):
    plt.plot(X[i], label='%s ' % (i+1), color=colors[i,:])
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  plt.xlabel("Days")
  plt.ylabel("Closing Price")
  plt.show()

  #Statistical tests
def statisticaltests(s):
  print('GRU and Baseline (RMSE)')
  U1, p = mannwhitneyu(s[:,1],s[:,2], alternative = 'two-sided')
  print('U='+ str(U1) + '. p = ' + str(p)) 
  print('GRU and Baseline (DA)')
  U1, p = mannwhitneyu(s[:,1+3],s[:,2+3], alternative = 'two-sided')
  print('U='+ str(U1) + '. p = ' + str(p))

if experiment ==1:
    #Retrieve BANKEX dataset 
    DATAX = genfromtxt('gru_examples/BANKEX.csv', delimiter=',')
else:
    #Retrieve Activities dataset 
    DATAX = genfromtxt('gru_examples/activities.csv', delimiter=',')

plot_series(DATAX)

#Normalize data between 0 and 1
DATAX, mima = scaleit(DATAX)
plot_series(DATAX)

selected_series = 0 #Select one signal arbitrarily to train the dataset
scaled_data = DATAX[selected_series, :] 
scaled_data = np.reshape(scaled_data, (len(scaled_data),1)) 
scaled_data.shape

#w < N < Q
window = 60 #Size of the window in days
test_samples = 251 #Number of test samples
N = len(scaled_data) - test_samples - window

trainX, trainY, testX, testY, X = data_preparation(window, scaled_data, N,f)

gru_model = GRU_Model(f)
epochs = 200

# Train GRU model
gru_trained = gru_model.fit(trainX, trainY, shuffle=True, epochs=epochs)

#Testing
s=(10,6) 
s=np.ones(s)  #for the results

for j in range(0,10):
  scaled_data = DATAX[j, :] #Changes the index of time_series
  scaled_data = np.reshape(scaled_data, (len(scaled_data),1)) 
  scaled_data.shape
  _ , _ , testX, testY, _ = data_preparation(window, scaled_data, N, f)
  y_pred_gru = gru_model.predict(testX)
  y_baseline = baselinef(testX,f) 
  testY = np.reshape(testY, (testY.shape[0],testY.shape[1]))
  print(testY.shape)
  s[j,1] = np.sqrt(mean_squared_error(testY, y_pred_gru))
  s[j,2] = np.sqrt(mean_squared_error(testY, y_baseline))
  s[j,1+3] = mda(testY, y_pred_gru)
  s[j,2+3] = mda(testY, y_baseline)


print('Mean values')
np.mean(s, axis=0)

print('Standard Deviation')
np.std(s, axis=0)

print('Statistical tests')
statisticaltests(s)

#use this code to save the models
if experiment == 1:
  ex = 'B'
else:
  ex = 'A'
gru_model.save(thepath+'/GRU_'+str(ex)+str(f)) #Saves GRU

#Rescale for visual inspection
testY = rescaleit(testY,mima,j)
y_pred_gru = rescaleit(y_pred_gru,mima,j)
y_baseline = rescaleit(y_baseline,mima,j)

if f == 1: 
  plt.plot(testY[0:100], label = 'Actual')
  plt.plot(y_baseline[0:100], label = 'Baseline prediction')
  plt.plot(y_pred_gru[0:100], label = 'GRU prediction')
  
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
  #plt.legend()
  plt.xlabel("Days")
  if experiment ==1:
    plt.ylabel("Closing Price")
  else:
    plt.ylabel("Value")
  plt.show()

g = 230 #Chooses one of the tests samples
if f == 20:
  plt.plot(testY[g,:], label = 'Actual')
  plt.plot(y_baseline[g,:], label = 'Baseline prediction')
  plt.plot(y_pred_gru[g,:], label = 'GRU prediction')
  plt.legend()
  plt.xlabel('Days')
  days = np.arange(testY.shape[1]+1)
  new_list = range(math.floor(min(days)), math.ceil(max(days))+1)
  plt.xticks(new_list)
  plt.ylabel('Value')
  plt.show()


#"""Plot loss function and Predictions"""
plt.plot(gru_trained.history['loss'], label = 'GRU los')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()