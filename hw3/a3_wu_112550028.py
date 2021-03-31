# Tim Wu SBUID: 112550028
# CSE354, Spring 2021
# Assignment 3

import torch
from torch import nn
import sys
import re
import numpy as np

# my system locale's character encoding is different, hence the need to explicitly mention utf-8.
sys.stdout = open('a2_wu_112550028.txt', 'w', encoding='utf-8') 

def getTargetWord(context):
    headMatch=re.compile(r'<head>([^<]+)</head>') #matches contents of head     
    tokens = context.split() #get the tokens

    headIndex = -1 #will be set to the index of the target word

    for i in range(len(tokens)):

        m = headMatch.match(tokens[i])

        if m: #a match: we are at the target token

            tokens[i] = m.groups()[0]

            headIndex = i

    return headIndex

def readData(file):
    # senses dict maps senses to unique integers
    # counts maps words to their frequencies.
    # process/machine/language store their respective lines of data.

    data = {"senses": {}, "counts": {}, "lemmas":[]}
    sensecounter = 0
    with open(file, mode="r", encoding='utf-8') as f:
        for line in f.readlines():

            curline = line.split("\t")
            lineID = curline[0]

            curcontext = curline[2]

            headIndex = getTargetWord(curcontext)

            # update word counts
            for word in curcontext.split(" "):
                split = word.split("/")
                token = split[0].lower()
                if "<head>" in token:
                    token = token[6:]
                if token not in data["counts"].keys():
                    data["counts"][token] = 0
                    
                data["counts"][token] = data["counts"][token] + 1
            
            # Remove lemma and POS
            finalcontext = '<s>'
            i = 0
            for word in curcontext.split(" "):
                token = word.split("/")[0].lower()
                if "<head>" in token:
                    token = token[6:]
                finalcontext += token + " "

            finalcontext = finalcontext.strip()
            
            # check lemma to properly classify the tuple we're going to create.
            lemma = curline[0].split(".")[0]

            # Assign an integer to this sense if we haven't seen it yet in the current lemma.
            sense = curline[1]
            if lemma not in data["senses"].keys():
                data["senses"][lemma] = {}

            if sense not in data["senses"][lemma].keys():
                data["senses"][lemma][sense] = sensecounter % 6
                sensecounter += 1

            # if a list for this lemma does not exist, we need to make it.
            # list of tuples that goes (headIndex, context, sense (string rep), unique context ID)
            if lemma not in data.keys():
                data[lemma] = [] # create list for lemma
                data["lemmas"].append(lemma) # also add it to the list of lemmas.

            finalcontext += "</s>"
            data[lemma].append((headIndex, finalcontext, sense, lineID))

        return data

def main():

    traindata = readData(sys.argv[1]) # use to train

    debug = True
    if debug:
        for part in traindata:
            print("{0}: ".format(part))
            print(traindata[part])
            print()

    # Create list of 5000 most frequent words.
    countsSorted = sorted(traindata["counts"].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    countsSorted = countsSorted[:5000] # we only want the 5000 most frequent ones.
    countsSorted = [tup[0] for tup in countsSorted]

if __name__ == "__main__":
    main()