import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

train = pd.read_csv('C:\Users\HP\Downloads\Recurrent_Neural_Networks\Google_Stock_Price_Train')
training_set = train.iloc[:, 1:2].values           ## This will still include only one column but it will convert it into numpy array.


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(training_set)

# Timesteps = 60  Output = 1
x_train = []                        #Will contain data from last 60 days
y_train = []                        # Will contain data 

#Calculating for each entry from 60 to 1258
for i in range(60, 1258):
    x_train.append(training_scaled(i - 60:i, 0))
    y_train.append(training_scaled(i,0))

##Converting to Numpy Arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the LSTM
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer 
regressor.add(LSTM(units = 50))         # return_sequences is False in the last layer
regressor.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)





    









