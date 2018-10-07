# The Arti Canon: Neural Text Generation
---
![articanon-gif]('./figures/buddha_gif.gif')
>         "Some quote from articanon here"
>                         - A Computer

Articanon is a neural text generation project designed to generate believable writing based on classic texts from Buddhism. It processes text from the Dhammapada as well as two different translations of the Bodhicaryvatara (see references for links to the originals)

#### A brief overview of the process:
1. `data.txt_to_npy.py` parses raw .txt copies of the original texts into a numpy datset consisting of (*x*,*y*) pairs of *sequence of characters*, *next character*. This matrix data is saved to the data directory.
2. `train.py` builds a sequence to sequence language model in Keras, then trains it on the text data, saving the weights to the model_saves directory.
3. `articanon.py` contains wrappers and utility functions for the text generation process.
4. `book.py` handles conversion from the raw .txt output of the generator to a pdf.
5. `write.py` is the main script for generating multiple chapters and assembling them into one 'book'/pdf.

**(A full technical overview of the project can be found [here](link to the medium post))**

##### Using the trained model to write your own book.
You can generate your own version of the articanon with the following command:  
```python write.py --k *int* --chapters *int* --verses *int*```
Where **k** is the width of the generator's beam search, **chapters** is the number of chapters to generate and **verses** is the number of psuedo-intellectual verses you'd like in each chapter.

There are a number of settings to tweak at the top of the Articanon class, the most obvious of which have been placed at the top of the file for convenience.

Note that the core capabilities of this library are located in the model training and generation scripts we've included. The `write.py` script in particular handles all the key interactions with most of the code base; that is the process that we prioritized debugging and getting to work. This is all just a fancy way of saying that, if you start to venture out from the default options-- like switching from beam search to temperature sampling or modifying the pdf output  -- you will probably start running into cases we didn't consider pretty quickly. However, most methods have docstrings and the pydoc files have been included (here)[articanon.Articanon.html] for convenience.
