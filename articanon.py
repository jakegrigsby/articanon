"""
Jake Grigsby
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
    
    Articanon(model, enlightened=True)
    
    --model: a keras-based language Model object.
    --enlightened: boolean that turns on advanced generation features like verse-by-verse beam search, spell checking and grammar.
    """

    def __init__(self, model=None, enlightened=True):
        f = open('data/full_text.txt','r')
        full_text = parse_raw_txt('data/full_text.txt')
        self.alphabet = sorted(list(set(full_text)))
        self.alph_idxs = dict((symbol, self.alphabet.index(symbol)) for symbol in self.alphabet)
        self.seq_len = 60
        self.model = model
        self.spellchecker = SpellChecker()
        self.spellchecker.word_frequency.load_text(full_text) #load domain specific words from training data
        self.beam = enlightened

    def data_tester(self):
        data = np.load('data/data.npz')
        x_seqs, y_chars = data['x'], data['y']
        r_subset = np.random.choice(100000, 5)
        for i, sentence in enumerate(x_seqs[r_subset]):
            print([self.idx2char(np.argmax(x)) for x in sentence])
            print(self.idx2char(np.argmax(y_chars[r_subset][i])))
        
    def generate_chapter(self, **kwargs):
        """
        Gate function that redirects towards the appropriate generation mechanism.
        self.englightened -> beam search generation
        !self.enlightened -> vanilla generation (softmax w/ variable temperature).
        """
        try:
            if self.beam:
                self._generate_chapter_beam(kwargs['k'], kwargs['nb_verse'], kwargs['output_path'], kwargs['seed'])
            else:
                self._generate_chapter_vanilla(kwargs['nb_verse'], kwargs['temperature'], kwargs['output_path'], kwargs['seed'])
        except KeyError:
            print("Invalid argument for generation type " + "BEAM" if self.beam else "VANILLA")

    def write_book(self, nb_chapters, **kwargs):
        """
        Generate the whole book in one function call!
        --nb_chapters: number of chapters in the book.
        --kwargs: parameters for corresponding generation method.
        if self.enlightened:
            kwargs = (k, nb_verse, output_path, seed)
               -k beam serach width
               -nb_verse is verses/chapter
               -output_path is output location of generated .txt files
               -seed is the initial input for the generation loop. None is fine, but first verse will likely be garbage.
                A solid buddha quote or fortune cookie works well.
        else:
            kwargs = (nb_verse, temperature, output_path, seed)
                -nb_vser is verses/chapter
                -temperature is softmax temperature for next-character selection. Temperature is directly proportional to generation
                creativity, inversely proporitonal to accuracy/spelling.
                -output_path is output location of generated .txt files
                -seed is the initial input for the generation loop. None is fine, but first verse will likely be garbage.
                A solid buddha quote or fortune cookie works well.
        """
        book = Book()
        book.set_title("Articanon")
        book.set_author('CavML')
        book.print_book_cover()
        for i, chapter in enumerate(range(nb_chapters)):
            self.generate_chapter(self.model, kwargs, output_path = 'chapter{}.txt'.format(i))
            book.print_chapter(i, self._new_chap_name(), 'chapter{}.txt'.format(i))
        book.output('articanon.pdf','F')

    def assemble_book(self, chapter_list):
        """
        Utility function for assembling pre-generated chapters into a pdf.
        --chapter_list = list of .txt generation file names outputed by generate_chapter() or similar generation algs.
                         In format of [['filename.txt', 'chapter name'], ['filename2.txt', '2nd chapter name'], etc]
        """
        book = Book()
        book.set_title("Articanon")
        book.set_author('CavML')
        book.print_book_cover()
        chap_num = 0
        title_of_chap = "Wisdom"
        for chapter in chapter_list:
            book.print_chapter(chap_num, chapter[1], chapter[0])
            chap_num += 1
        book.output('articanon.pdf', 'F')

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

    def _clean_raw_output(self, input_text, output_path='output/clean_output.txt'):
        """
        Clean up raw output from generator. Saves to a txt file (output_path).
        """
        output_text_file = open(output_path, 'w')
        input_text = input_text.split('1')
        output_text = ''
        for num, verse in enumerate(input_text):
            output_text += "{}. {}".format(num + 1, verse) + '\n\n'
        output_text = self.editor(output_text)
        output_text_file.write(output_text)

    def _generate_chapter_vanilla(self, nb_verse=300, temperature=.3, output_path = 'output/vanilla_output.txt', seed=None):
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
                try:
                    seed = text[-self.seq_len:]
                except IndexError:
                    seed = text
                seed = seed.replace('1','')
                x = self.string2matrix(seed, self.seq_len)
                preds = self.model.predict(x)[0]
                next_idx = self._sample(preds, temperature)
                next_char = self.idx2char(next_idx)
                text += next_char
                if next_char == '.' and i > 200: #arbitrary 'soft stopping'
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

    def _generate_chapter_beam(self, k, nb_verse, output_path, seed):
        """
        verse by verse beam search. Includes:
            - Repetition reduction
            - final re-scoring

        Takes a while to run. You can keep an eye on the progress using the keras-style progress bar.

        Generates in format of '###. verse text here\n'. That output is automatically cleaned by _clean_raw_output()
        """
        score = lambda y, y_1: (y  + np.log(y_1))
        text = ''
        if seed != None:
            seed = seed[:self.seq_len]
            text += seed.lower()
        progbar = Progbar(nb_verse)
        for verse in range(nb_verse):
            try:
                seed = text[-self.seq_len:]
            except IndexError:
                seed = text
            hypotheses = [(seed, 0.)]
            running = True
            while running:
                new_hypotheses = []
                terminated = 0
                for h in hypotheses:
                    if (h[0][-1] == '.' and len(h[0]) > 150) or len(h[0]) > 650: #this branch has terminated
                        new_hypotheses.append(h)
                        terminated += 1
                        continue
                    x = self.string2matrix(h[0], self.seq_len)
                    k_best_chars, k_best_probs = self.k_best(k, self.model.predict(x)[0])
                    for i, char in enumerate(k_best_chars):
                        s = score(h[1], k_best_probs[i])
                        #reduce repetition
                        s = s - 100000000000 if (h[0][-10:]+char in h[0]) else s
                        new_hypotheses.append((h[0]+char, s))
                if terminated <= k:
                    hypotheses = sorted(new_hypotheses, key=itemgetter(1))[-k:]
                running = False if terminated == k else True
            hypotheses = [self._final_score(h) for h in hypotheses]
            best = sorted(hypotheses, key=itemgetter(1))[-1]
            text += "1" +  best[0]
            progbar.update(verse + 1)
        if seed != None:
           text = text[len(seed):] 
        self._clean_raw_output(text, output_path=output_path)

    def editor(self, text):
        """TODO
        """
        text = re.sub(r'[a-z]([.?!]) ([a-z])', r'\g<1> \g<2>'.capitalize(), text)
        return text

    def _final_score(self, hypothesis):
        """
        Pick between the top k final hypotheses according to a more detailed scoring function.
        """
        string, score = hypothesis[0], hypothesis[1]
        #length normalization
        score /= len(string)**.5
        #better vocabulary, longer sentences
        words = string.split(' ')
        unique_words = len(set(words))
        score += .5*unique_words +.2*len(words)
        #spelling
        misspelled = self.spellchecker.unknown(words)
        score -= 5*len(misspelled)

        return (string, score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', choices=['vanilla','beam'],default='beam')
    args = parser.parse_args()
    model = train.build_model()
    model.load_weights('model_saves/articanon.h5')
    generator = Articanon(model)
    if args.ver == 'beam':
        generator.generate_chapter(nb_verse=2, k=2, output_path='output/first_chap_output.txt', seed="In truth, all of life is suffering. What we do not know, we")
        generator.assemble_book([['output/first_chap_output.txt','Hello']])
    if args.ver == 'vanilla':
        generator.beam = False
        generator.generate_chapter(nb_verse=3, temperature=.4, output_path='output/first_chap_output.txt', seed='He who lives looking for pleasures only, his senses uncontrolled, immoderate in his food')
    generator.assemble_book([['output/first_chap_output.txt', 'Enlightenment']])
