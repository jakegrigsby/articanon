"""
References:
https://www.tibethouse.jp/about/buddhism/text/pdfs/Bodhisattvas_way_English.pdf
http://promienie.net/images/dharma/books/shantideva_way-of-bodhisattva.pdf
https://www.buddhanet.net/pdf_file/scrndhamma.pdf
"""
import re
import numpy as np

def parse_raw_txt(source='full_text.txt'):
    f = open(source,'r')
    full_text = f.read()
    full_text = re.sub(r'[0-9]+.?\s?','21',full_text).lower() #delete verse counts, replace with start char (1)
    full_text = re.sub(r'1\n','21',full_text) # add in end char (2)
    full_text = re.sub(r'\n','',full_text)
    full_text = re.sub(r'[\‘\’\`\'\"\—\"\”\ʹ\ʺ]?','\"',full_text) #who knew there were so many ways to use quotes?
    full_text = re.sub(r'[\-•]?','',full_text)
    f.close()
    return full_text

if __name__ == "__main__":
    full_text = parse_raw_txt('full_text.txt')
    X_LEN = 50
    STRIDE = 2

    x_seqs = []
    y_chars = []

    # Model will take in a sequence of length X_LEN and learn to predict the next character (teacher forcing)
    for t in range(0, len(full_text) - X_LEN, STRIDE):
        x_seqs.append(full_text[t:t+X_LEN])
        y_chars.append(full_text[t+X_LEN])

    alphabet = sorted(list(set(full_text)))
    print(len(alphabet))
    alph_idxs = dict((symbol, alphabet.index(symbol)) for symbol in alphabet)
    print(alph_idxs)
    if __name__ == '__main__':
        # We are going to build a one hot encoded matrix of shape (# of seqs), (length of seq), (num of possible characters)
        x_matrix = []
        for seq in x_seqs:
            chars = np.array([alph_idxs[char] for char in seq]).astype('int')
            one_hot_chars = np.eye(len(alphabet))[chars]
            x_matrix.append(one_hot_chars)
        x_matrix = np.array(x_matrix)
        assert x_matrix.shape == (len(x_seqs), X_LEN, len(alphabet))

        # Next we build the one hot encoded matrix of shape (# of seqs), (num of possible characters)
        y_chars_idxs = np.array([alph_idxs[char] for char in y_chars])
        y_matrix = np.eye(len(alphabet))[y_chars_idxs]
        assert y_matrix.shape == (len(x_seqs), len(alphabet))

        # Save the data to a numpy archive format
        np.savez('data.npz', x=x_matrix, y=y_matrix)
