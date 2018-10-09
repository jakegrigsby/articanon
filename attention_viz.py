"""
Cavalier Machine Learning, University of Virginia
September 2018

This is a script to visualize attention matrices for the input defined in ex_string.
The writeup uses a fictional example, but the diagram this outputs looks very similar.
"""
from train import build_model
import numpy as np
from articanon import Articanon
import keras.backend as K
import matplotlib.pyplot as plt

model = build_model()
model.load_weights('model_saves/articanon_best.h5f')
articanon = Articanon(model)

ex_string = "The Buddha declared in unequivocal terms that consciousness depends on matter, sensation, perception and mental formations and that it cannot exist independently of them."
ex_string = "It must be repeated here that according to Buddhist philosophy there is no permanent, unchanging spirit which can be considered"
ex_string = "To the seeker after Truth it is immaterial from where an idea comes. The source".lower()
ex_x = [articanon.string2matrix(ex_string, 60)]

attention_outputs = model.layers[-5].output[0]
attention_viz = K.function(inputs=[model.input], outputs=[attention_outputs])
attention_vals = attention_viz(ex_x)
attention_vals = np.array(attention_vals)


print(attention_vals.shape)

fig = plt.imshow(attention_vals[:,:,0], cmap='Oranges', aspect=5)
fig.axes.get_yaxis().set_visible(False)
fig.axes.get_xaxis().set_visible(False)
plt.savefig('figures/attention_viz.png')
