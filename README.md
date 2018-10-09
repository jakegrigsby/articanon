# The Arti Canon: Neural Text Generation

<img src='https://media.giphy.com/media/jnUMLiKLwxuPdnji4J/giphy.gif' style='width: 100%'>

>         "Why should I be angry with the childish pain of conditioned existence?"
>                          - A Computer

Arti Canon is a neural text generation project designed to generate believable writing based on classic texts from Buddhism. It processes text from *The Dhammapada*, *Zen Mind, Beginner's Mind*, as well as two different translations of *The Bodhicaryvatara* (see references for links to the originals).

#### A brief overview of the process:
1. `data/txt_to_npy.py` parses raw .txt copies of the original texts into a numpy datset consisting of (*x*,*y*) pairs of (*sequence of characters*, *next character*). This array is saved to the data directory.
2. `train.py` builds a sequence to sequence language model in Keras, then trains it on the text data, saving the weights to the model_saves directory.
3. `articanon.py` contains wrappers and utility functions for the text generation process.
4. `book.py` handles conversion from the raw .txt output of the generator to a pdf.
5. `write.py` is the main script for generating multiple chapters and assembling them into one 'book'/pdf.

**(A full technical overview of the project can be found [here](link to the medium post))**

##### Using the trained model to write your own book
You can generate your own version of the articanon with the following command:  
```bash
 write.py --chapters *int* --verses *int* --k *int*
```
Where **k** is the width of the generator's beam search, **chapters** is the number of chapters to generate and **verses** is the number of pseudo-intellectual verses you'd like in each chapter.

There are a number of settings to tweak at the top of the `Articanon` class, the most obvious of which are set up as properties of each `Articanon` object.

Note that the core capabilities of this library are located in the model training and generation scripts we've included. The `write.py` script in particular handles all the key interactions with most of the code base; that is the process that we prioritized debugging and getting to work. This is all just a fancy way of saying that, if you start to venture out from the default options-- like switching from beam search to temperature sampling or modifying the pdf output  -- you will probably start running into cases we didn't consider pretty quickly. However, most methods have docstrings and the pydoc files have been included [here](documentation.html) for convenience.

###### That's cool, but what if I just want to run the entire project myself?
I don't know why you'd want to do this, but it took for me 4 seconds to write and will take your laptop 4 days to run-- that's a time efficiency opportunity that was too good to pass up:  
```bash
chmod u+x articanon.sh
./articanon.sh
```

##### References
1. [*The Bodhicaryvatara* v1](https://www.tibethouse.jp/about/buddhism/text/pdfs/Bodhisattvas_way_English.pdf)  
2. [*The Bodhicaryvatara* v2](http://promienie.net/images/dharma/books/shantideva_way-of-bodhisattva.pdf)  
3. [*The Dhammapada*](https://www.buddhanet.net/pdf_file/scrndhamma.pdf)  
4. [Zen Mind Beginner's Mind](https://archive.org/stream/ZenMindBeginnersMind-ShunruyuSuzuki/zenmind_djvu.txt)
5. [*Deep Learning with Python*](https://www.manning.com/books/deep-learning-with-python)  
6. [Neural Text Generation, a Practical Guide](https://arxiv.org/abs/1711.09534)
