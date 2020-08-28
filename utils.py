import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import gc

kfcv_seed = 1998
kfold_func = StratifiedKFold
data_enhance_method = []
k = 5

def set_data_enhance(val):
    if not isinstance(val, list):
        val = [val]
    global data_enhance_method
    data_enhance_method = val

mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
    4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
    8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
    12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
    16: 'C_2', 17: 'C_5', 18: 'C_6'}

reversed_mapping = {value: key for key, value in mapping.items()}

def decode_label(label_code):
    str = mapping[label_code]
    scene_code = ord(str.split('_')[0]) - ord('A')
    action_code = ord(str.split('_')[1]) - ord('0')
    return scene_code, action_code

def kfcv_evaluate(model_name, x, y):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    evals = {'loss':0.0, 'accuracy':0.0}
    index = 0

    for train, val in kfold.split(x, np.argmax(y, axis=-1)):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        
        model = keras.models.load_model('./models/%s/part_%d.h5' % (model_name, index))

        loss, acc = model.evaluate(x=x[val], y=y[val])
        evals['loss'] += loss / k
        evals['accuracy'] += acc / k
        index += 1
    return evals

def kfcv_predict(model_name, inputs):
    path = './models/' + model_name + '/'
    models = []
    for i in range(k):
        models.append(keras.models.load_model(path + 'part_%d.h5' % i))

    print('%s loaded.' % model_name)
    result = []
    for j in range(k):
        result.append(models[j].predict(inputs))

    print('result got')
    result = sum(result) / k
    return result

def kfcv_fit(builder, x, y, epochs, checkpoint_path, verbose=2, batch_size=64):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    histories = []
    evals = []

    if checkpoint_path[len(checkpoint_path) - 1] != '/':
        checkpoint_path += '/'

    for i in range(k):
        if os.path.exists(checkpoint_path + 'part_%d.h5' % i):
            os.remove(checkpoint_path + 'part_%d.h5' % i)

    for index, (train, val) in enumerate(kfold.split(x, np.argmax(y, axis=-1))):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        model = builder()

        x_train = x[train]
        y_train = y[train]

        if len(data_enhance_method) > 0:
            x_train_copy = np.copy(x_train)
            y_train_copy = np.copy(y_train)
            for method in data_enhance_method:
                x_, y_ = data_enhance(method, x_train_copy, y_train_copy)
                x_train = np.r_[x_train, x_]
                y_train = np.r_[y_train, y_]
            x_train, y_train = shuffle(x_train, y_train)
            print('Data enhanced (%s) => %d' % (' '.join(data_enhance_method), len(x_train)))

        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path + 'part_%d.h5' % index,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

        h = model.fit(x=x_train, y=y_train,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x[val], y[val]),
                callbacks=[checkpoint],
                batch_size=batch_size,
                shuffle=True
                )
        evals.append(model.evaluate(x=x[val], y=y[val]))
        histories.append(h)
        del model
        gc.collect()
    return histories, evals

def data_enhance(method, train_data, train_labels):
    if method == 'noise':
        noise = train_data + np.random.normal(0, 0.1, size=train_data.shape)
        return noise, train_labels
    
    elif method == 'mixup':
        index = [i for i in range(len(train_labels))]
        np.random.shuffle(index)

        x_mixup = np.zeros(train_data.shape)
        y_mixup = np.zeros(train_labels.shape)

        for i in range(len(train_labels)):
            x1 = train_data[i]
            x2 = train_data[index[i]]
            y1 = train_labels[i]
            y2 = train_labels[index[i]]

            factor = np.random.beta(0.2, 0.2)

            x_mixup[i] = x1 * factor + x2 * (1 - factor)
            y_mixup[i] = y1 * factor + y2 * (1 - factor)

        return x_mixup, y_mixup

def save_results(path, output):
    print('saving...')

    df_r = pd.DataFrame(columns=['fragment_id', 'behavior_id'])
    for i in range(len(output)):
        behavior_id = output[i]
        df_r = df_r.append(
            {'fragment_id': i, 'behavior_id': behavior_id}, ignore_index=True)
    df_r.to_csv(path, index=False)

def infer(model_name, inputs, csv_output):
    output = np.argmax(kfcv_predict(model_name, inputs), axis=-1)
    save_results(csv_output, output)
    print('- END -')
    print('Your file locates at %s' % csv_output)

def shuffle(data, labels, seed=None):
    index = [i for i in range(len(labels))]
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(index)
    return data[index], labels[index]