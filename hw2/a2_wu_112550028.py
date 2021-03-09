# Tim Wu SBUID: 112550028
# CSE354, Spring 2021
# Assignment 2

import torch
import sys
import re
import numpy

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

    context = ' '.join(tokens)
    return headIndex, context

def readData(file):
    # senses dict maps senses to unique integers
    # counts maps words to their frequencies.
    # process/machine/language store their respective lines of data.

    data = {"senses": {}, "counts": {}, "lemmas":[]}
    sensecounter = 0
    with open(file, mode="r", encoding='utf-8') as f:
        for line in f.readlines():
            curline = line.split("\t")

            # Assign an integer to each sense
            sense = curline[1]
            if sense not in data["senses"].keys():
                data["senses"][sense] = sensecounter
                sensecounter += 1

            curcontext = ''
            curcontext = curcontext.join(curline[2:]) # Anything beyond index 2 is the context.

            headIndex, curcontext = getTargetWord(curcontext)

            # update word counts
            for word in curcontext.split(" "):
                split = word.split("/")
                if split[2] == "PUNCT":
                    continue
                elif split[0].lower() not in data["counts"]:
                    data["counts"][split[0].lower()] = 1
                else:
                    data["counts"][split[0].lower()] = data["counts"][split[0].lower()] + 1
            
            # Remove lemma and POS
            finalcontext = ''
            for word in curcontext.split(" "):
                if word.split("/")[2] != "PUNCT":
                    finalcontext = finalcontext + " "

                # ignore punctuation
                if word.split("/")[2] == "PUNCT":
                    continue
                finalcontext += word.split("/")[0]

            finalcontext = finalcontext.strip()
            

            # check lemma to properly classify as a tuple of (headIndex, finalContext).
            lemma = curline[0].split(".")[0]
            # if a list for this lemma does not exist, we need to make it.
            if lemma not in data.keys():
                data[lemma] = [] # create list for lemma
                data["lemmas"].append(lemma) # also add it to the list of lemmas.

            data[lemma].append((headIndex, finalcontext))
        return data

def createOneHots(contextTup: tuple, word):
    context = contextTup[1]
    index = contextTup[0]
    split = context.split(" ")

    before = [0] * len(split)
    # avoid out of bounds array indexing. if the keyword is at index 0 or neg, can't look before. if in
    if index > 0 and index < len(split) and split[index - 1] == word:
        before[index - 1] = 1
        before[index] = 1

    after = [0] * len(split)
    # if index is at the last valid index (len - 1), then can't look after.
    if index + 1 < len(split) and split[index + 1]:
        after[index + 1] = 1
        after[index] = 1
    
    return before, after

def main():

    data = readData(sys.argv[1])
    # debugging
    debug = True
    if debug:
        for part in data:
            print("{0}: ".format(part))
            print(data[part])
            print()

    onehots = []
    # for each context in each lemma, we wanna create 2 one-hots for every word.
    if (len(data["counts"]) <= 2000):
        for lemma in data["lemmas"]:  # loop through all our lemmas
            for contextTup in data[lemma]: # loop through all the contexts for the current lemma
                for word in data["counts"].keys: # for each word in our vocabulary, we want to create two one-hots.
                    before, after = createOneHots(contextTup, word)
                    #print(before)
                    #print(after)

    else:
        countsSorted = sorted(data["counts"], key=data["counts"].get, reverse=True) # sort (we lose the counts in the process but that's fine)
        for lemma in data["lemmas"]:  # loop through all our lemmas
            for contextTup in data[lemma]: # loop through all the contexts for the current lemma
                for i in range(2000): # for each word in our vocabulary, we want to create two one-hots.
                    word = countsSorted[i]
                    before, after = createOneHots(contextTup, word)
                    print(before)
                    print(after)
                    return
    

if __name__ == '__main__':
    main()