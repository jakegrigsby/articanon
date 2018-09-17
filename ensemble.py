from keras.models import Model
from keras.layers.core import *
from keras.layers import Input, Bidirectional, LSTM, Multiply, Concatenate
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from snapshot import Snapshot
import numpy as np
from keras.utils import plot_model
from train import nb_cycles

BATCH_SIZE = 128
X_LEN = 50
alphabet_len = 45
hidden_dim = 350

def build_static_model(cycle_num):
    encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True))(x)
    attention = Dense(1, activation='tanh')(encoder)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*hidden_dim)(attention)
    attention = Permute([2, 1])(attention)
    attention = Multiply()([encoder, attention])
    decoder = LSTM(hidden_dim)(attention)
    y = Dense(alphabet_len, activation='softmax')(decoder)
    model = Model(inputs=x, outputs=y)
    for layer in model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[categorical_accuracy, 'acc'])
    model.load_weights('model_saves/weights_cycle_{}.h5'.format(cycle_num))
    return model

x = Input((X_LEN, alphabet_len))
m1 = build_static_model('%02d' % nb_cycles - 3)(x)
m2 = build_static_model('%02d' % nb_cycles - 2)(x)
m3 = build_static_model('%02d' % nb_cycles - 1)(x)
m4 = build_static_model('%02d' % nb_cycles)(x)
concat = Concatenate(axis=1)([m1,m2,m3,m4])
dense = Dense(264)(concat)
dense2 = Dense(128)(dense)
y = Dense(alphabet_len, activation='softmax')(dense2)
model = Model(inputs=x, outputs=y)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[categorical_accuracy, 'acc'])

if __name__ == "__main__":
    data = np.load('data/data.npz')
    permutation = np.load('validation_split.npy')
    x_seqs, y_chars = data['x'][permutation], data['y'][permutation]
    assert x_seqs.shape[0] == y_chars.shape[0]

    plot_model(model, show_shapes=True, to_file='model_saves/ensemble.png')

    callbacks = [
        EarlyStopping(patience=3),
        ReduceLROnPlateau(patience=1),
    ]

    history = model.fit(x_seqs, y_chars, batch_size=128, epochs=50, callbacks=callbacks, validation_split=.3)
    model.save('model_saves/articanon.h5')
