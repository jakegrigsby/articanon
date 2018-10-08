from train import build_model
import numpy as np
from articanon import Articanon
import keras.backend as K
import matplotlib.pyplot as plt

model = build_model()
model.load_weights('model_saves/articanon_best.h5f')
articanon = Articanon(model)

ex_string = "if you see an intelligent man who tells you where true treasures are to be found, who shows what is to be avoided,"
ex_x = [articanon.string2matrix(ex_string, 60)]

attention_outputs = model.layers[-4].output[0]
attention_viz = K.function(inputs=[model.input], outputs=[attention_outputs])
attention_vals = attention_viz(ex_x)
attention_vals = np.array(attention_vals)


print(attention_vals.shape)

fig = plt.imshow(attention_vals[:,:,0], cmap='Oranges', aspect=5)
fig.axes.get_yaxis().set_visible(False)
fig.axes.get_xaxis().set_visible(False)
plt.savefig('figures/attention_viz.png')
