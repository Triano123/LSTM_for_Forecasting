import pandas as pd 
import numpy as np 

#visualization
import seaborn as sns 
import matplotlib.pyplot as plt

#module deep learning 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM

#ploting with keras
from livelossplot import PlotLossesKeras

#math module
import math
from math import sqrt

#save model 
from keras.models import load_model


#fix random seed for reproducibility
np.random.seed(7)

#import data yahoo finance
df = pd.read_csv('yahoo_finance.csv')
df.head()

#setting index as date
df['Date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
df.index = df['Date']
df.head()

#visualization of the data using maplotlib
def plot(columns_date, columns_numeric, title = None):
    if title is None :
        title = 'Time Series Plot'
    # plot time series     
    figure = plt.figure(figsize = (12,6))
    plt.plot(df[columns_date], df[columns_numeric], label = 'Actual', color = 'blue')
    plt.legend(loc=4)
    plt.title(title, fontsize = 15)
    plt.show()

# input parameter to the module
# plot time series 2  
plot('Date', 'Close')

#choose columns date and close variable
data = df[['Date','Close']]

#new dataframe 
new_df = pd.DataFrame(index = range(0,len(df)), columns = ['Date','Close'])
#looping in range 
for i in range(0, len(data)):
    new_df['Date'][i] = data['Date'][i]
    new_df['Close'][i] = data['Close'][i]

#setting index
new_df.index = new_df.Date
new_df.drop('Date', axis=1, inplace=True)

#Creating data train and testing 
dataset = new_df.values
dataset

# since the lstm sensitive to the scale data
## so we will change the data to the same scale by use min max scaler
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#train and test
#in order to train dataset we take the data from 2010 till 2017
train = dataset[0:2014,:] 
#in order to test dataset we take the data that is 2018
test = dataset[2014:,:]

# split a univariate sequence into samples
def split_sequence(df, look_back):
    '''
    parameter   :
    df          : Object
        dataframe that would be analyzed 
    look_back   :  int
        the number of period in order to look back for foreacst in the future 
    '''
    dataX, dataY = list(), list()
    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + look_back
        # check if we are beyond the sequence
        if end_ix > len(df)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = df[i:end_ix,0], df[end_ix,0]
        dataX.append(seq_x)
        dataY.append(seq_y)
    return np.array(dataX), np.array(dataY)

#train test split
look_back = 2
trainX, trainY = split_sequence(train,look_back)
testX, testY = split_sequence(test, look_back)

# reshape input to be [samples, time steps, features]
time_step = 1
trainX = np.reshape(trainX, (trainX.shape[0], time_step, trainX.shape[1]))
testX  = np.reshape(testX, (testX.shape[0], time_step, testX.shape[1]))
testX.shape[0]


#creating model LSTM 
model = Sequential()
model.add(LSTM(4 ,input_shape = (time_step, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainX, trainY,
        validation_data = (testX, testY),
        epochs = 100, 
        batch_size= 1, 
        verbose= 2, 
        callbacks=[PlotLossesKeras()])


#save model keras 
save_model = model.save('model.h5')

#load model keras
model = load_model('model.h5')

#make prediction 
pred_train = model_1.predict(trainX)
pred_test = model_1.predict(testX)

#invert prediction to real value
##train
pred_train = scaler.inverse_transform(pred_train)
trainY = scaler.inverse_transform([trainY])
## test
pred_test = scaler.inverse_transform(pred_test)
testY = scaler.inverse_transform([testY])

## MODEL EVALUATION USING RMSE AND MAPE
def model_eval_forecasting(train_Y, test_Y, pred_train, pred_test, eval = None) :
    if eval is None :
        eval == 'rmse' 
    
    #Calculate root means square errors (RMSE)
    train_rmse = math.sqrt(mean_squared_error(train_Y, pred_train))
    test_rmse = math.sqrt(mean_squared_error(test_Y, pred_test))
    
    #Calculate mean absolute percentage error (MAPE)
    train_mape = np.mean(np.abs((trainY-pred_train)/trainY)) 
    test_mape = np.mean(np.abs((testY-pred_test)/testY))
    
    if eval is 'rmse' :
        #print 
        print('Model Evaluation')
        print('RMSE train & test :' , train_rmse, '&', test_rmse)
    elif eval is 'mape' :
        #print 
        print('Model Evaluation :')
        print('MAPE train & test :', train_mape,'%','&',test_mape,'%')
    elif (eval == 'rmse') and (eval == 'mape') :
        #Model evaluate 
        print('\nModel Evaluate Time Series Analytics = ')
        print('1. Root Mean Squared Error (RMSE)')
        print('RMSE Train & Test = ', train_rmse, "&",test_rmse)
        print('\n2.Mean Percentage Absolute Error = ')
        print('MAPE Train & Test = ', train_mape,'%','&',test_mape,'%')

#use the function
model_eval_forecasting(trainY, testY, pred_train, pred_test, eval = ('rmse','mape')

#Visualization 2  
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(pred_train)+look_back] = pred_train
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
testPredictPlot[len(pred_train)+(look_back*1)+1:len(dataset)-1, :] = pred_test

# plot baseline and predictions
plt.figure(figsize=(12,6))
plt.plot(dataset, label = 'Actual Dataset')
plt.plot(trainPredictPlot, label = 'train predict')
plt.plot(testPredictPlot, label='test predict ', color = 'red' )
plt.title('Comparison actual and predict')
plt.legend(loc= 4)
plt.show()


#############################################
#          Predict for new data             #
#############################################

#predict for new data 
new_data = dataset[-4:]
new_data

#setting how many look back that we used
look_back = 2
data_x, data_y= split_sequence(new_data, look_back)

#result 
# demonstrate prediction
time_step = 1
data_x = data_x.reshape((1,look_back,time_step))
yhat = model.predict(data_x)
yhat
