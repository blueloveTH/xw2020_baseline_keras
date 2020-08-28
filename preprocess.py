import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 特征列名称
src_names = ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg', 'acc', 'acc_g']

def handle_features(data):
    data.drop(columns=['time_point'], inplace=True)

    data['acc'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** 0.5
    data['acc_g'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** 0.5

    return data

# 构造numpy特征矩阵
def handle_mats(grouped_data):
    mats = [i.values for i in grouped_data]
    # padding
    for i in range(len(mats)):
        padding_times = 61 - mats[i].shape[0]
        for j in range(padding_times):
            mats[i] = np.append(mats[i], [[0 for _ in range(mats[i].shape[1])]], axis=0)

    mats_padded = np.zeros([len(mats), 61, mats[0].shape[1]])
    for i in range(len(mats)):
        mats_padded[i] = mats[i]

    return mats_padded

def get_test_data(use_scaler=True):
    FILE_NAME = "dataset/sensor_test.csv"
    data = handle_features(pd.read_csv(FILE_NAME))
    if use_scaler:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        data[src_names] = scaler.transform(data[src_names].values)

    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]
    return handle_mats(grouped_data)

def get_train_data(use_scaler=True, shuffle=True, pseudo_labels_file=None):
    df = pd.read_csv("dataset/sensor_train.csv")

    # 简单拼接伪标签
    if pseudo_labels_file != None:
        df = df.append(pd.read_csv(pseudo_labels_file))
    data = handle_features(df)

    # 标准化，并将统计值保存
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(data[src_names].values)  
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        data[src_names] = scaler.transform(data[src_names].values)

    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]
    train_labels = np.array([int(i.iloc[0]['behavior_id']) for i in grouped_data])
    for i in range(len(grouped_data)):
        grouped_data[i].drop(columns='behavior_id', inplace=True)
    train_data = handle_mats(grouped_data)
    
    if shuffle:
        index = [i for i in range(len(train_labels))]
        np.random.seed(2020)
        np.random.shuffle(index)

        train_data = train_data[index]
        train_labels = train_labels[index]

    return train_data, train_labels

def get_train_test_data(use_scaler=True, shuffle=True, pseudo_labels_file=None):
    train_data, train_lables = get_train_data(use_scaler, shuffle, pseudo_labels_file=pseudo_labels_file)
    test_data = get_test_data(use_scaler)
    return train_data, train_lables, test_data