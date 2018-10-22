"""
Cavalier Machine Learning, University of Virginia
September 2018

Script for repairing books that have already been generated
(for implementing changes to the pdf structure or editor).
"""
from argparse import ArgumentParser
from articanon import Articanon

parser = ArgumentParser()
parser.add_argument('--chapters', default=40, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    articanon = Articanon()
    chapter_list = []
    for chap in range(args.chapters):
        filename = './output/chapter{}.txt'.format(chap+1)
        x = open(filename, 'r')
        verses = x.readlines()
        x.close()
        repaired = ''
        for verse in verses:
            repaired += articanon.editor(verse) + '\n'
        repaired_file = './output/chapter{}_repaired.txt'.format(chap+1)
        y = open(repaired_file, 'w+')
        y.write(repaired)
        y.close()
        chapter_list.append((repaired_file, articanon.new_chapter_title()))

    articanon.assemble_book(chapter_list, output_path='output/articanon_repaired.pdf')
