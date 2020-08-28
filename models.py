import tensorflow as tf
import numpy as np
from tensorflow import keras

def BLOCK(seq, filters, kernal_size):
    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(seq)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, kernal_size, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)

    seq = keras.layers.Conv1D(filters, 1)(seq)
    seq = keras.layers.Add()([seq, cnn])
    return seq

def BLOCK2(seq, filters=128, kernal_size=5):
    seq = BLOCK(seq, filters, kernal_size)
    seq = keras.layers.MaxPooling1D(2)(seq)
    seq = keras.layers.SpatialDropout1D(0.3)(seq)
    seq = BLOCK(seq, filters//2, kernal_size)
    seq = keras.layers.GlobalAveragePooling1D()(seq)
    return seq

def ComplexConv1D(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape[1:])
    seq_3 = BLOCK2(inputs, kernal_size=3)
    seq_5 = BLOCK2(inputs, kernal_size=5)
    seq_7 = BLOCK2(inputs, kernal_size=7)
    seq = keras.layers.concatenate([seq_3, seq_5, seq_7])
    seq = keras.layers.Dense(512, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    seq = keras.layers.Dense(128, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(seq)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=tf.optimizers.Adam(1e-3),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.1),           
            metrics=['accuracy'])

    return model