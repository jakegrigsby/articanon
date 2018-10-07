"""
Cavalier Machine Learning, University of Virginia
September 2018
"""
from keras.models import Model
from keras.layers.core import *
from keras.layers import Input, Bidirectional, LSTM, Multiply, Embedding
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.regularizers import l2
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from keras.utils import plot_model

BATCH_SIZE = 128
X_LEN = 60
ALPHABET_LEN = 40
HIDDEN_DIM = 350
NB_CYCLES = 12
EPOCHS = 600

def random_shuffle(x, y):
    permutation = np.random.permutation(x.shape[0])
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def build_model(seq_len=X_LEN):
    x = Input((seq_len, ALPHABET_LEN))
    encoder = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)
    encoder = LSTM(HIDDEN_DIM, return_sequences=True, recurrent_dropout=.3, dropout=.2)(encoder)
    attention = Dense(1, activation='tanh')(encoder)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(HIDDEN_DIM)(attention)
    attention = Permute([2, 1])(attention)
    attention = Multiply()([encoder, attention])
    decoder = LSTM(HIDDEN_DIM, recurrent_dropout = .2, dropout= .15, return_sequences=True)(attention)
    decoder = LSTM(HIDDEN_DIM, recurrent_dropout=.25, dropout=.2)(decoder)
    y = Dense(ALPHABET_LEN, activation='softmax')(decoder)
    model = Model(inputs=x, outputs=y)
    return model

if __name__ == "__main__":
    data = np.load('data/data.npz')
    x_seqs, y_chars = random_shuffle(data['x'], data['y'])
    assert x_seqs.shape[0] == y_chars.shape[0]

    model = build_model()
    model.compile(optimizer=Adam(.001), loss='categorical_crossentropy', metrics=[categorical_accuracy, 'acc'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_saves/model.png')

    callbacks = [
        TensorBoard(log_dir='log_dir/'),
        ModelCheckpoint('model_saves/articanon_best.h5f', period=2, monitor='acc', save_best_only=True),
        ModelCheckpoint('model_saves/articanon_latest.h5f', period=2, save_best_only=False)
    ]

    history = model.fit(x_seqs, y_chars, batch_size=512, epochs=EPOCHS, callbacks=callbacks, validation_split=0, shuffle=True)
