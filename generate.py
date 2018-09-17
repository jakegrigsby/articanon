from train import build_model
import numpy as np
import keras
import re
from ensemble import model
from operator import itemgetter
from keras.utils import Progbar
from data.txt_to_np import parse_raw_txt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', choices=['vanilla','beam'],default='beam')
args = parser.parse_args()

def char2vec(char):
    return np.eye(len(alphabet))[alph_idxs[char]]

def idx2char(idx):
    global alphabet
    return alphabet[idx]

def string2matrix(string, max_len):
    string = string[-max_len:]
    matrix = np.zeros((1, X_LEN, len(alphabet)))
    for i, char in enumerate(string):
        vec = char2vec(char)
        matrix[:,i,:] = vec
    return matrix

def k_best(k, prob_vec):
    k_best_idxs = np.argsort(prob_vec)[-k:]
    k_best_chars = [idx2char(idx) for idx in k_best_idxs]
    k_best_probs = [prob_vec[idx] for idx in k_best_idxs]
    return k_best_chars, k_best_probs

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

def beamsearch_generate(k, model, nb_verse, output_path='output/raw_output_beamsearch.txt', seed=None):
    global X_LEN, alphabet

    score = lambda y, y_1: (y  + np.log(y_1))

    output = open(output_path,'w')
    text = '1'
    if seed != None:
        text += seed.lower()
    progbar = Progbar(nb_verse)
    for verse in range(nb_verse):
        try:
            seed = text[-50:]
        except IndexError:
            seed = text
        hypotheses = [(seed, 0.)]
        running = True
        while running:
            new_hypotheses = []
            terminated = 0
            for h in hypotheses:
                if h[0][-1] == '2' or (h[0][-1] == '.' and len(h[0]) > 250) or len(h[0]) > 350: #this branch has terminated
                    terminated += 1
                    continue
                x = string2matrix(h[0], X_LEN)
                k_best_chars, k_best_probs = k_best(k, model.predict(x)[0])
                for i, char in enumerate(k_best_chars):
                    new_hypotheses.append((h[0]+char, score(h[1], k_best_probs[i])))
            if terminated < k:
                hypotheses = sorted(new_hypotheses, key=itemgetter(1))[-k:]
            #print(hypotheses)
            running = False if terminated == k else True
        best = sorted(hypotheses, key=itemgetter(1))[-1]
        print(best)
        text += best[0]
        text += '1'
        progbar.update(verse + 1)
    output.write(text)
    output.close()

    clean_raw_output(input_path=output_path, output_path='output/clean_output_beamsearch.txt')

if __name__ == "__main__":

    f = open('data/full_text.txt','r')
    full_text = parse_raw_txt()
    alphabet = sorted(list(set(full_text)))
    alph_idxs = dict((symbol, alphabet.index(symbol)) for symbol in alphabet)
    X_LEN = 50

    model.load_weights('model_saves/articanon.h5')

    if args.mode == 'beam':
        beamsearch_generate(5, model, nb_verse=300, seed='All that we are is the result of what we have thought: it is')
    if args.mode == 'vanilla':
        vanilla_generate(model, nb_verse=300, temperature=.4)
