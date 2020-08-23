# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:51:23 2018

@author: Zannatul Ferdoush
"""

import math
import time

import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing

df = pd.read_csv("../stock_prices_bd_2008_2017.csv", index_col=0)
df1 = pd.read_csv("../stock_prices_bd_2008_2017.csv", skiprows=2, delimiter=',',
                  names=['date', 'trading_code', 'opening_price', 'closing_price', 'low', 'high', 'volume'])

df["adj close"] = df.closing_price  # Moving close to the last column
df.drop(['closing_price'], 1, inplace=True)  # Moving close to the last column
print(df.head())

trading_code = list(set(df.trading_code))
print(len(trading_code))
print(trading_code[:11])  # Example of what is in trading_code

df = df[df.trading_code == 'GLAXOSMITH']
df.drop(['trading_code'], 1, inplace=True)
print(df.head())

# plotting of date vs midprice plot
df1 = df1[df1.trading_code == 'GLAXOSMITH']
df1.drop(['trading_code'], 1, inplace=True)
print(df1.head())

df1 = df1.sort_values('date')

plt2.figure(figsize=(18, 9))
plt2.plot(range(df1.shape[0]), (df1['low'] + df1['high']) / 2.0)  # df.shape[0])= row count
plt2.xticks(range(0, df1.shape[0], 500), df1['date'].loc[::500], rotation=45)
plt2.xlabel('Date', fontsize=18)
plt2.ylabel('Mid Price', fontsize=18)
plt2.savefig('../LSTM_assets/GLAXOSMITH_mid_vs_date.png', bbox_inches='tight')
plt2.show()


def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['opening_price'] = min_max_scaler.fit_transform(df.opening_price.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
    return df


df = normalize_data(df)
print(df.head())


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)  # 5
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split
    train = result[:int(row), :]  # 90% date, all features

    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    d = 0.3
    model = Sequential()

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

    # adam = keras.optimizers.Adam(decay=0.2)
    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


window = 22  # not clear, have to ask sir
X_train, y_train, X_test, y_test = load_data(df, window)
print(X_train[0], y_train[0])

model = build_model([5, window, 1])

model.fit(X_train, y_train, batch_size=512, epochs=90, validation_split=0.1, verbose=1)

# print(X_test[-1])
diff = []
ratio = []
p = model.predict(X_test)
print(p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))


# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value):
    df = df['adj close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


newp = denormalize(df, p)
newy_test = denormalize(df, y_test)


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)

plt2.plot(newp, color='red', label='Prediction')
plt2.plot(newy_test, color='blue', label='Actual')
plt2.legend(loc='best')
plt2.savefig('../LSTM_assets/GLAXOSMITH_LSTM_predicted_result.png', bbox_inches='tight')
plt2.show()
