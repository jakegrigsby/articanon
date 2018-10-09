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
    The main wrapper class for generating text. Uses a language model ( f(x) = p(x_n|x_0...x_n-1) ) to generate text verse by verse.
    Verses are compiled into a pdf, using book.Book.

    --model: a keras-based language Model object.
    --repetition_penalty: score penalty for repetition during beam search.
    --spell_penalty: score penalty for misspelled words. Calculated during final evaluation of beam search.
    --unique_words_reward: score reward for unique words, encouraging larger vocabulary. Calculated during final evaluation of beam search.
    --length_normalization_alpha: exponent for length normalization of beam search hypotheses. Calculated during final evaluation of beam search.
    """
    def __init__(self, model=None, repetition_penalty=5, spell_penalty=10, unique_words_reward=.55, length_normalization_alpha=.3,):
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
        self.repetition_penalty = repetition_penalty
        self.spell_penalty = spell_penalty
        self.unique_words_reward = unique_words_reward
        self.length_normalization_alpha = length_normalization_alpha

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
        --string: input string.
        --max_len: size of fixed output vector. Shorter inputs are padded with zeros, longer inputs are trimmed.
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
        --preds: vector output of a softmax layer.
        --temp: temperature for sampling function. Higher values increase selection of less-likely characters.
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
        f = open(text_path, 'r')
        input_text = f.read()
        f.close()
        verses = input_text.split('\n\n')
        num = 1
        text = ''
        for verse in verses[:-1]:
            print(verse)
            accept = input("Accept this verse? (y/n) ")
            if accept.strip() == 'y':
                verse = re.sub(r'[0-9]+\.','',verse)
                text += "{}. {}".format(num, verse) + '\n\n'
                num += 1
        f = open(text_path, 'w')
        f.write(text)
        f.close()

    def _clean_raw_output(self, input_text, delete_first=False, output_path='output/clean_output.txt'):
        """
        Clean up raw output from generator. Saves to a txt file (output_path).
        --delete_first: Delete the first verse. Used when generating multiple chapters
        and giving the generator a seed for each one is too tedious; allows for more
        randomness and 'self inspiration'.
        --output_path: where to save the cleaned txt file.
        """
        output_text_file = open(output_path, 'w')
        input_text = input_text.split('1')
        input_text = input_text[1:] if delete_first else input_text
        output_text = ''
        num = 0
        for verse in input_text:
            output_text += self.editor("{}. {}".format(num + 1, verse.strip())) + '\n\n'
            num += 1
        output_text_file.write(output_text)
        output_text_file.close()

    def generate_chapter_vanilla(self, nb_verse=30, temperature=.3, output_path = 'output/vanilla_output.txt', seed=None):
        """
        Generates text using repeated single-character prediction loop. Samples from language model's output distribution according to some temperature.
        --nb_verse: number of verses to generate.
        --temperature: temperature for sampling function. Higher values increase selection of less-likely characters.
        --output_path: where to output raw txt file containing generated text.
        --seed: Starting seed. This vanilla version doesn't include the delete_first/random_seed approach of beam search.
        """
        text = ''
        if seed != None:
            seed = seed[:self.seq_len]
            text += seed.lower()
        for verse in range(nb_verse):
            verse_len = 0
            for i in range(1000): #prevent infinite loops if verse never wants to end...
                seed = text.replace('1','')
                try:
                    seed = seed[-self.seq_len:]
                except IndexError:
                    seed = seed
                x = self.string2matrix(seed, self.seq_len)
                preds = self.model.predict(x)[0]
                next_idx = self._sample(preds, temperature)
                next_char = self.idx2char(next_idx)
                verse_len += 1
                text += next_char
                if re.match(r'[.?)!]', text[-1]) and verse_len > 90: #arbitrary 'soft stopping'
                    break
            text += "1"
        self._clean_raw_output(text, output_path=output_path, delete_first=False)

    def k_best(self, k, prob_vec):
        """
        Return k-best hypotheses.
        --k: beam search width
        --prob_vec: vector of character probabilities.
        """
        k_best_idxs = np.argsort(prob_vec)[-k:]
        k_best_chars = [self.idx2char(idx) for idx in k_best_idxs]
        k_best_probs = [prob_vec[idx] for idx in k_best_idxs]
        return k_best_chars, k_best_probs

    @property
    def random_seed(self):
        """
        new random seed from the source text. used to seed first verse of each chapter,
        which is usually deleted.
        """
        rand_start = np.random.randint(0, len(self.full_text))
        return self.full_text[rand_start:rand_start+self.seq_len]

    def generate_chapter_beam(self, k, nb_verse, output_path, delete_first, seed=None):
        """
        verse by verse beam search. Includes:
            - Repetition reduction
            - final re-scoring

        Takes a while to run. You can keep an eye on the progress using the keras-style progress bar.

        Generates in format of '###. verse text here\n'. That output is automatically cleaned by _clean_raw_output()
        --k: beam search width. The tree search will clip itself after the *k* best hypotheses. Very effective between 1-10,
        with diminishing return after that. Heavily impacts runtime.
        --nb_verse: number of verses to generate.
        --delete_first: Delete the first verse. Used when generating multiple chapters
        and giving the generator a seed for each one is too tedious; allows for more
        randomness and 'self inspiration'.
        --seed: optional starting seed for text generation loop-if you want to prompt
        articanon to talk about something specific. If none is given, a random seed is taken
        from the source text and is used to write the first verse, which is then discarded.
        """
        if delete_first:
            nb_verse += 1
            seed = self.random_seed
        #lambda function for accumlated sentence score
        score = lambda y, y_1: (y  + np.log(y_1))
        text = seed[:self.seq_len].lower()
        #keras-style progress bar
        progbar = Progbar(nb_verse)
        for verse in range(nb_verse):
            seed = text.replace('1','') #1 is used to split verses but isn't part of the model's vocabulary
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
                             s -= self.repetition_penalty
                        new_hypotheses.append((h[0]+char, s))
                if terminated <= k:
                    hypotheses = sorted(new_hypotheses, key=itemgetter(1))[-k:] #consider the k best options next iteration
                running = False if terminated >= k else True
            #print(sorted(hypotheses, key=itemgetter(1)))
            hypotheses = [self._final_score(h) for h in hypotheses]
            #print(sorted(hypotheses, key=itemgetter(1)))
            best = sorted(hypotheses, key=itemgetter(1))[-1]
            best = best[0][len(seed):] + "1" #'1' added for splitting
            text += best
            progbar.update(verse + 1)
        self._clean_raw_output(text[:-1], delete_first=delete_first, output_path=output_path)

    def editor(self, text):
        """
        The dataset is too small to use capitalized letters, so the grammar the model learns has no concept of capitalization. This converts that learned grammar to proper English.
        Or we're just cheating.
        --text: input text.
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
        """
        return 'random' chapter title from shuffled list of Dhammapada chapter titles.
        If all titles have been used, you've read so much Buddhism you've reached enlightenment;
        titles will be called Awakening.
        """
        if self.titles:
            return self.titles.pop()
        return "Awakening"

    def _final_score(self, hypothesis):
        """
        Pick between the top k final hypotheses according to a more detailed scoring function.
        """
        string, score = hypothesis[0], hypothesis[1]
        #length normalization
        score /= len(string)**self.length_normalization_alpha
        #better vocabulary, longer sentences
        words = string.split(' ')
        for i, word in enumerate(words):
            words[i] = re.sub(r'[,\.?1\]\[:;\)\(]','',word)
        #print(words)
        unique_words = len(set(words))
        #print(unique_words)
        score += self.unique_words_reward*unique_words
        #spelling
        misspelled = self.spellchecker.unknown(words)
        score -= self.spell_penalty*len(misspelled)
        return (string, score)

if __name__ == "__main__":
    """
    Debugging script. Actual book is generated/assembled in write.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', choices=['vanilla','beam'],default='beam')
    args = parser.parse_args()
    model = train.build_model()
    model.load_weights('model_saves/articanon_best.h5f')
    generator = Articanon(model)
    if args.ver == 'beam':
        generator.generate_chapter_beam(nb_verse=1, k=15, output_path='output/first_chap_output.txt', seed='what is the runtime complexity of a doubly linked list with ghost nodes?', delete_first=False)
        generator.filter_verses('output/first_chap_output.txt')
    if args.ver == 'vanilla':
        generator.generate_chapter_vanilla(nb_verse=1, temperature=.8, output_path='output/first_chap_output.txt', seed='Let a wise man blow off the impurities of his self, as a smith blows off the impurities of silver')
    generator.assemble_book([['output/first_chap_output.txt', 'Enlightenment']])
