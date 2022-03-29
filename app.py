import pandas as pd
import numpy as np
from numpy import array
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, TimeDistributed, Bidirectional
import re

from sklearn               import metrics
from sklearn.preprocessing import MinMaxScaler

from scipy.ndimage         import gaussian_filter1d
from scipy.signal          import medfilt
import tensorflow as tf
from numpy.random          import seed
seed(1)
tf.random.set_seed(1)

from keras_radam import RAdam


def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        
        if end_ix > len(sequence)-1:
            break
            
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y[0])
    return array(X), array(y)

def data_process(data, sc):
    train_x = pd.read_csv(data, usecols=['日期', '備轉容量(MW)'])

    day = 2
    for i, x in enumerate(train_x['日期']):
        # index = holiday.loc[holiday['date'] == x].index
        # if holiday['isHoliday'][index].values == '是':
        #     train_x.loc[i, 'isHoliday'] = 1
        # else:
        #     train_x.loc[i, 'isHoliday'] = 0
        s = train_x.loc[i, '日期'].split('/')
        if len(s[1]) > 1:
            if len(s[2]) > 1:
                train_x.loc[i, '日期'] = s[0] + s[1] + s[2]
            else:
                train_x.loc[i, '日期'] = s[0] + s[1] + str(0) + s[2]
        else:
            if len(s[2]) > 1:
                train_x.loc[i, '日期'] = s[0] + str(0) + s[1] + s[2]
            else:
                train_x.loc[i, '日期'] = s[0] + str(0) + s[1] + str(0) + s[2]
                
        train_x.loc[i, '日期'] = float(str(train_x.loc[i, '日期'])[4:])

        train_x.loc[i, '備轉容量(MW)'] = train_x.loc[i, '備轉容量(MW)'] * 10
        train_x.loc[i, 'day'] = day
        day = day + 1
        if day > 8:
            day = 1
            train_x.loc[i, 'day'] = day
            day = day + 1
    
    train_x['備轉容量(MW)'] = medfilt(train_x['備轉容量(MW)'], 3)              
    train_x['備轉容量(MW)'] = gaussian_filter1d(train_x['備轉容量(MW)'], 1.2)

    train_set    = train_x.reset_index(drop=True).drop(['日期', ],axis=1)
    training_set = train_set.values

    a = train_set.iloc[:, 0:1].values

    training_set = training_set.T


    training_set[0] = sc.fit_transform(a).reshape(-1)
    training_set = training_set.T

    X_train, y_train = data_split(training_set, 365)
    X_train          = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
    # X_test, y_test   = data_split(testing_set, 75)
    # X_test           = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)


    return X_train, y_train, training_set
        
def get_model():

    model = Sequential()
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True),
                                input_shape=(X_train.shape[1], 2)))
    model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Dense(17))
    model.compile(RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-6 ,clipnorm=1.0), loss='mean_squared_error')
    
    return model


def train_model(model, X_train, y_train):
    X_train = K.cast_to_floatx(X_train)

    history = model.fit(X_train, y_train, epochs=1000, batch_size=256)
    loss    = history.history['loss']


if __name__ == '__main__':
    config=tf.compat.v1.ConfigProto() 
    config.gpu_options.visible_device_list = '0' 
    config.gpu_options.allow_growth = True 
    sess=tf.compat.v1.Session(config=config)


    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    sc = MinMaxScaler(feature_range=(0, 1))
    X_train, y_train, training_set = data_process(args.training, sc)
    model = get_model()
    train_model(model, X_train, y_train)
    result = pd.DataFrame(['20220330', '20220331', '20220401', '20220402', '20220403', '20220404', '20220405', '20220406', '20220407', '20220408', '20220409', '20220410', '20220411', '20220412', '20220413'], columns=['date'])
    result['operating_reserve(MW)'] = sc.inverse_transform(model.predict(training_set[-365:].reshape(1, 365, 2))).ravel()[2:].reshape(-1)
    result['operating_reserve(MW)'] = result['operating_reserve(MW)'].astype(int)
    print('----------Output File------------')


    result.to_csv('submission.csv',index=False)

    # 近三年每日尖峰備轉容量率.csv