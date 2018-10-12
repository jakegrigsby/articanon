"""
Cavalier Machine Learning, University of Virginia
September 2018

Script for training articanon's language model.
Uses encoder/decoder seq2seq model built with LSTMs and attention.
Bidirectional first layer, two layer encoder and two layer decoder.
Dropout is applied to help with generalization, but in this use case
we aren't too concerned with overfitting. Categorical cross-entropy 
loss with an adam optimizer.

Uses tensorboard for visualization during the relatively long (600 epoch)
training run. Takes ~44 hours on a single mid-tier GPU.

Saves best (according to training set accuracy; we aren't using validation
data) and latest weights to ./model_saves.
"""
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from keras.utils import plot_model
from articanon import Articanon

BATCH_SIZE = 512
X_LEN = 60
ALPHABET_LEN = 35
HIDDEN_DIM = 350
EPOCHS = 600

if __name__ == "__main__":
    data = np.load('data/data.npz')
    x_seqs, y_chars = data['x'], data['y']
    assert x_seqs.shape[0] == y_chars.shape[0]
    articanon = Articanon()
    model = articanon.model
    model.compile(optimizer=Adam(.001), loss='categorical_crossentropy', metrics=[categorical_accuracy, 'acc'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_saves/model.png')

    callbacks = [
        TensorBoard(log_dir='log_dir/'),
        ModelCheckpoint('model_saves/articanon_best.h5f', period=2, monitor='acc', save_best_only=True),
        ModelCheckpoint('model_saves/articanon_latest.h5f', period=2, save_best_only=False)
    ]

    history = model.fit(x_seqs, y_chars, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_split=0, shuffle=True)
