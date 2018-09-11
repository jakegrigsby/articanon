from train import build_model
import numpy as np
import keras
import re
from ensemble import model

f = open('data/full_text.txt','r')
full_text = f.read()
full_text = re.sub(r'[0-9]+. ','1',full_text).lower() #delete verse counts, replace with start char (1)
full_text = re.sub(r'\n1','2',full_text) # add in end char (2)
alphabet = sorted(list(set(full_text)))
alph_idxs = dict((symbol, alphabet.index(symbol)) for symbol in alphabet)
X_LEN = 50

model.load_weights('model_saves/articanon.h5')

def char2vec(char):
    return np.eye(len(alphabet))[alph_idxs[char]]

def idx2char(idx):
    global alphabet
    return alphabet[idx]

def string2matrix(string, max_len):
    string = string[:max_len]
    matrix = np.zeros((1, X_LEN, len(alphabet)))
    for i, char in enumerate(string):
        vec = char2vec(char)
        matrix[:,i,:] = vec
    return matrix

def k_best(k, prob_vec):
    k_best_idxs = np.argsort(prob_vec)[-k:]
    k_best_chars = [idx2char(idx) for idx in k_best_idxs]
    k_best_probs = [prob_vec[idx] for idx in k_best_idxs]
    return zip(k_best_chars, k_best_probs)

def _sample(preds, temp):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def vanilla_generate(model, output_path='output/raw_output_vanilla.txt', nb_verse=300, temperature=.3, seed=None):
    global X_LEN, alphabet
    output = open(output_path,'w')
    text = ''
    if seed != None:
        text += seed.lower()
    for verse in range(nb_verse):
        text += '1'
        for i in range(1000): #prevent infinite loops if verse never wants to end...
            try:
                seed = text[-50:]
            except IndexError:
                seed = text
            x = string2matrix(seed, X_LEN)
            preds = model.predict(x)[0]
            next_idx = _sample(preds, temperature)
            next_char = idx2char(next_idx)
            text += next_char
            if next_char == '2':
                break
    output.write(text)
    output.close()

    clean_raw_output(input_path=output_path, output_path='output/clean_output_vanilla.txt')

def clean_raw_output(input_path, output_path='output/clean_output.txt'):
    input_text_file = open(input_path, 'r')
    input_text = input_text_file.read()
    input_text_file.close()

    output_text_file = open(output_path, 'w')
    input_text = input_text[1:].split('1')
    output_text = ''
    for num, verse in enumerate(input_text):
        verse = verse.replace('1','')
        verse = verse.replace('2','\n')
        print("{}. {}".format(num + 1, verse))
        output_text += "{}. {}".format(num + 1, verse)
    output_text_file.write(output_text)

vanilla_generate(model, temperature=.6, nb_verse=10, seed='We are shaped by our thoughts; we become what we think. When the mind is pure, joy follows like a shadow that never leaves.')


# def beamsearch(k, seed, model):
#     hypotheses = [(seed, 0.)]
#
#
#
# output = open('raw_output.txt','x')
# output.write('1')
#
# seed = output[-min(len(output),50):]
# for verse in range(400):
#     output.write(beamsearch(3, seed, model))
