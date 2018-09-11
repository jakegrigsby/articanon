import tensorflow as tf
from keras.models import Model
from keras.layers.core import *
from keras.layers import Input, Bidirectional, LSTM, Multiply
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard
from snapshot import Snapshot
import numpy as np
from keras.utils import plot_model

BATCH_SIZE = 128
X_LEN = 50
alphabet_len = 44
hidden_dim = 350

data = np.load('data/data.npz')
x_seqs, y_chars = data['x'], data['y']
assert x_seqs.shape[0] == y_chars.shape[0]

def build_model(seq_len=X_LEN):
    x = Input((seq_len, alphabet_len))
    encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True, recurrent_dropout=.35, dropout=.2))(x)
    attention = Dense(1, activation='tanh')(encoder)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(2*hidden_dim)(attention)
    attention = Permute([2, 1])(attention)
    attention = Multiply()([encoder, attention])
    decoder = LSTM(hidden_dim, recurrent_dropout=.35, dropout=.2)(attention)
    y = Dense(alphabet_len, activation='softmax')(decoder)
    model = Model(inputs=x, outputs=y)
    return model

if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer=RMSprop(lr=.002), loss='categorical_crossentropy', metrics=[categorical_accuracy, 'acc'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_saves/model.png')

    callbacks = [
        Snapshot('model_saves',nb_epochs=500,nb_cycles=18,verbose=1),
        TensorBoard(log_dir='log_dir')
    ]

    history = model.fit(x_seqs, y_chars, batch_size=128, epochs=300, callbacks=callbacks, validation_split=.3)
