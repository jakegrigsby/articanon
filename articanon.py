"""
Cavalier Machine Learning, University of Virginia
September 2018
"""
from train import build_model
import numpy as np
import keras
import re
import train
from  operator import itemgetter
from keras.utils import Progbar
from data.txt_to_np import parse_raw_txt
import argparse
from book import Book
from spellchecker import SpellChecker

class Articanon:
    """
    The main wrapper class for generating text. Uses a language model ( p(y|x_0...x_n) ) to generate text verse by verse.
    Verses are compiled into a pdf, using book.Book.

    --model: a keras-based language Model object.
    """
    def __init__(self, model=None):
        f = open('data/full_text.txt','r')
        self.full_text = parse_raw_txt('data/full_text.txt')
        f.close()
        self.alphabet = sorted(list(set(self.full_text)))
        self.alph_idxs = dict((symbol, self.alphabet.index(symbol)) for symbol in self.alphabet)
        self.seq_len = 60
        self.model = model
        self.spellchecker = SpellChecker()
        self.spellchecker.word_frequency.load_text(self.full_text) #load domain specific words from training data
        with open('data/title_names.txt','r') as f:
            self.titles = f.read().split('\n')
            np.random.shuffle(self.titles)

    def data_tester(self):
        data = np.load('data/data.npz')
        x_seqs, y_chars = data['x'], data['y']
        r_subset = np.random.choice(100000, 5)
        for i, sentence in enumerate(x_seqs[r_subset]):
            print([self.idx2char(np.argmax(x)) for x in sentence])
            print(self.idx2char(np.argmax(y_chars[r_subset][i])))

    def assemble_book(self, chapter_list):
        """
        Utility function for assembling pre-generated chapters into a pdf.
        --chapter_list = list of .txt generation file names outputed by generate_chapter() or similar generation algs.
                         In format of [['filename.txt', 'chapter name'], ['filename2.txt', '2nd chapter name'], etc]
        """
        book = Book()
        book.set_title("The Arti Canon")
        book.set_author('CavML')
        book.print_book_cover()
        chap_num = 1
        for chapter in chapter_list:
            book.print_chapter(chap_num, chapter[1], chapter[0])
            chap_num += 1
        book.output('output/articanon.pdf', 'F')

    def char2vec(self, char):
        """
        Convert ascii character to one_hot_vector, as determined by one hot idxs in txt_to_np.py
        """
        return np.eye(len(self.alphabet))[self.alph_idxs[char]]

    def idx2char(self, idx):
        """
        Convert index (taken from one hot encoded vector) to character, as determined by txt_to_np.py
        """
        return self.alphabet[idx]

    def string2matrix(self, string, max_len):
        """
        Convert ascii string to one_hot_encoded matrix, as determined by txt_to_np.py
        """
        string = string[-max_len:]
        matrix = np.zeros((1, self.seq_len, len(self.alphabet)))
        for i, char in enumerate(string):
            vec = self.char2vec(char)
            matrix[:,i,:] = vec
        return matrix

    def _sample(self, preds, temp):
        """
        Sampling from model's output with an adjusted distribution.
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        return np.argmax(probs)

    def filter_verses(self, text_path):
        """
        Simple routine for filtering out unwanted verses. Call after a chapter has been generated.
        --text_path: the raw output of the generation step.
        """
        input_text = open(text_path, 'r+')
        input_text = input_text.read()
        verses = input_text.split('\n\n')
        for num, verse in enumerate(verses[:-1]):
            print(verse)
            accept = input("Accept this verse? (y/n) ")
            if accept.strip() == 'y':
                verses[num] = re.sub(r'[0-9]*\.','',verse) + "1"
            else:
                verses[num] = ''
        text = ''
        for verse in verses:
            text += verse
        self._clean_raw_output(text[:-1], text_path) #cut the last '1' out

    def _clean_raw_output(self, input_text, delete_first=False, output_path='output/clean_output.txt', live_monitor=False):
        """
        Clean up raw output from generator. Saves to a txt file (output_path).
        """
        output_text_file = open(output_path, 'w')
        input_text = input_text.split('1')
        input_text = input_text[1:] if delete_first else input_text
        output_text = ''
        num = 0
        for verse in input_text:
            answer = 'y'
            if live_monitor:
                print(verse)
                answer = input("Accept verse? (y/n)" )
            if answer == 'y' or answer == 'yes' or answer == "Y":
                 output_text += self.editor("{}. {}".format(num + 1, verse)) + '\n\n'
                 num += 1
        output_text_file.write(output_text)

    def generate_chapter_vanilla(self, nb_verse=300, temperature=.3, output_path = 'output/vanilla_output.txt', seed=None):
        """
        Generates text using repeated single-character prediction loop. Samples from language model's output distribution according to some temperature.
        """
        text = ''
        if seed != None:
            seed = seed[:self.seq_len]
            text += seed.lower()
        for verse in range(nb_verse):
            text += '1'
            for i in range(1000): #prevent infinite loops if verse never wants to end...
                seed = seed.replace('1','')
                try:
                    seed = text[-self.seq_len:]
                except IndexError:
                    seed = text
                x = self.string2matrix(seed, self.seq_len)
                preds = self.model.predict(x)[0]
                next_idx = self._sample(preds, temperature)
                next_char = self.idx2char(next_idx)
                text += next_char
                if next_char == '.' and i > 150: #arbitrary 'soft stopping'
                    break
        self._clean_raw_output(text, output_path=output_path)

    def k_best(self, k, prob_vec):
        """
        Return k-best hypotheses.
        """
        k_best_idxs = np.argsort(prob_vec)[-k:]
        k_best_chars = [self.idx2char(idx) for idx in k_best_idxs]
        k_best_probs = [prob_vec[idx] for idx in k_best_idxs]
        return k_best_chars, k_best_probs

    @property
    def random_seed(self):
        rand_start = np.random.randint(0, len(self.full_text))
        return self.full_text[rand_start:rand_start+self.seq_len]

    def generate_chapter_beam(self, k, nb_verse, output_path, delete_first, live_monitor, seed=None):
        """
        verse by verse beam search. Includes:
            - Repetition reduction
            - final re-scoring

        Takes a while to run. You can keep an eye on the progress using the keras-style progress bar.

        Generates in format of '###. verse text here\n'. That output is automatically cleaned by _clean_raw_output()
        """
        if delete_first:
            nb_verse += 1
            seed = self.random_seed
        #lambda function for accumlated sentence score
        score = lambda y, y_1: (y  + np.log(y_1))
        text = seed[:self.seq_len].lower()
        progbar = Progbar(nb_verse)
        for verse in range(nb_verse):
            seed = text.replace('1','')
            try:
                seed = seed[-self.seq_len:]
            except IndexError:
                seed = seed
            hypotheses = [(seed, 0.)]
            running = True
            while running:
                new_hypotheses = []
                terminated = 0
                for h in hypotheses:
                    if re.match(r'[.?)!]', h[0][-1]) and len(h[0]) > 150: #this branch has terminated
                        new_hypotheses.append(h)
                        terminated += 1
                        continue
                    x = self.string2matrix(h[0], self.seq_len)
                    k_best_chars, k_best_probs = self.k_best(k, self.model.predict(x)[0])
                    for i, char in enumerate(k_best_chars):
                        s = score(h[1], k_best_probs[i])
                        #reduce repetition
                        if h[0][-15:]+char in h[0]:
                             s -= 50
                        new_hypotheses.append((h[0]+char, s))
                if terminated <= k:
                    hypotheses = sorted(new_hypotheses, key=itemgetter(1))[-k:]
                running = False if terminated >= k else True
            hypotheses = [self._final_score(h) for h in hypotheses]
            best = sorted(hypotheses, key=itemgetter(1))[-1]
            best = best[0][len(seed):] + "1"
            text += best
            progbar.update(verse + 1)
        self._clean_raw_output(text[:-1], delete_first=delete_first, output_path=output_path, live_monitor=live_monitor)

    def editor(self, text):
        """
        The dataset is too small to use capitalized letters, so the grammar the model learns has no concept of capitalization. This converts that learned grammar to proper English.

        Or we're just cheating.
        """
        #start of sentence capitalization
        sentences = re.findall(r'[^.!?]+[.!?]', text)
        text = ''
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = sentence.capitalize()
            text += sentence + ' '
        #'i' capitalization
        text = re.sub(r' i ', ' I ', text)
        text = re.sub(r' i, ', ' I, ', text)
        return text

    def new_chapter_title(self):
        return self.titles.pop()

    def _final_score(self, hypothesis):
        """
        Pick between the top k final hypotheses according to a more detailed scoring function.
        """
        string, score = hypothesis[0], hypothesis[1]
        #length normalization
        score /= len(string)
        #better vocabulary, longer sentences
        words = string.split(' ')
        for i, word in enumerate(words):
            words[i] = re.sub(r'[,\.?1:;\)\(]','',word)
        #print(words)
        unique_words = len(set(words))
        #print(unique_words)
        score += 5*unique_words
        #spelling
        misspelled = self.spellchecker.unknown(words)
        score -= 2000*len(misspelled)
        return (string, score)

if __name__ == "__main__":
    """
    Debugging script. Actual book is generated/assembled in write.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', choices=['vanilla','beam'],default='beam')
    args = parser.parse_args()
    model = train.build_model()
    model.load_weights('model_saves/articanon.h5f')
    generator = Articanon(model)
    if args.ver == 'beam':
        generator.generate_chapter_beam(nb_verse=3, k=5, output_path='output/first_chap_output.txt', seed="And he who lives a hundred years, idle and weak, a life of one day is better", live_monitor=False)
        generator.filter_verses('output/first_chap_output.txt')
    if args.ver == 'vanilla':
        generator.generate_chapter_vanilla(nb_verse=3, temperature=.4, output_path='output/first_chap_output.txt', seed='He who lives looking for pleasures only, his senses uncontrolled, immoderate in his food')
    generator.assemble_book([['output/first_chap_output.txt', 'Enlightenment']])
