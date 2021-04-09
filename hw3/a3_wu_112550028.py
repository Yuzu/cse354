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

    data = {"counts": {}, "contexts": []}

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
            
            # add <s> and </s> tokens
            if "<s>" not in data["counts"].keys():
                data["counts"]["<s>"] = 0

            if "</s>" not in data["counts"].keys():
                data["counts"]["</s>"] = 0
            
            data["counts"]["<s>"] = data["counts"]["<s>"] + 1
            data["counts"]["</s>"] = data["counts"]["</s>"] + 1

            # Remove lemma and POS
            finalcontext = '<s> '
            i = 0
            for word in curcontext.split(" "):
                token = word.split("/")[0].lower()
                if "<head>" in token:
                    token = token[6:]
                finalcontext += token + " "

            finalcontext = finalcontext.strip()
            
            finalcontext += " </s>"

            data["contexts"].append(finalcontext.split(" "))

        return data

# counts has the count of every single token in our corpus.
# vocab has only the 5000 most frequent tokens in our corpus.
def extractUnigramCounts(counts: dict, vocab: list):
    
    unigrams = {}
    unigrams["<OOV>"] = 0

    for key in counts.keys():
        if key not in vocab:
            unigrams["<OOV>"] += counts[key]
        else:
            unigrams[key] = counts[key]

    return unigrams

# Create the counts in a sparse dictionary of dictionaries.
def extractBigramCounts(contexts: list, vocab: list):
    counts = {}
    for context in contexts:
        for i in range(0, len(context) - 1):
            word1 = context[i]
            word2 = context[i + 1]
            
            # both oov
            if (word1 not in vocab) and (word2 not in vocab):
                counts.setdefault("<OOV>", {}) # add OOV for first word
                counts["<OOV>"].setdefault("<OOV>", 0) # add OOV for second word

                counts["<OOV>"]["<OOV>"] += 1
            
            # first word oov
            elif word1 not in vocab:
                counts.setdefault("<OOV>", {}) # add OOV for first word
                counts["<OOV>"].setdefault(word2, 0)

                counts["<OOV>"][word2] += 1

            #second word oov
            elif word2 not in vocab:
                counts.setdefault(word1, {})
                counts[word1].setdefault("<OOV>", 0) # add OOV for second word

                counts[word1]["<OOV>"] += 1

            # none oov
            else:
                counts.setdefault(word1, {})
                counts[word1].setdefault(word2, 0)
    
                counts[word1][word2] += 1

    return counts

def extractTrigramCounts(contexts: list, vocab: list):
    counts = {}
    for context in contexts:
        for i in range(0, len(context) - 2):
            word1 = context[i] 
            word2 = context[i + 1]
            word3 = context[i + 2]
            
            # [(oov, oov)][oov] - all out of vocab = 000
            if (word1 not in vocab) and (word2 not in vocab) and (word3 not in vocab):
                counts.setdefault(("<OOV>", "<OOV>"), {})
                counts[("<OOV>", "<OOV>")].setdefault("<OOV>", 0)

                counts[("<OOV>", "<OOV>")]["<OOV>"] += 1

            # [(oov, oov)][word3] = 001
            elif (word1 not in vocab) and (word2 not in vocab) and (word3 in vocab):
                counts.setdefault(("<OOV>", "<OOV>"), {})
                counts[("<OOV>", "<OOV>")].setdefault(word3, 0)

                counts[("<OOV>", "<OOV>")][word3] += 1

            # [(oov, word2)][oov] = 010
            elif (word1 not in vocab) and (word2 in vocab) and (word3 not in vocab):
                counts.setdefault(("<OOV>", word2), {})
                counts[("<OOV>", word2)].setdefault("<OOV>", 0)

                counts[("<OOV>", word2)]["<OOV>"] += 1

            # [(oov, word2)][word3] = 011
            elif (word1 not in vocab) and (word2 in vocab) and (word3 in vocab):
                counts.setdefault(("<OOV>", word2), {})
                counts[("<OOV>", word2)].setdefault(word3, 0)

                counts[("<OOV>", word2)][word3] += 1

            # [(word1, oov)][oov] = 100
            elif (word1 in vocab) and (word2 not in vocab) and (word3 not in vocab):
                counts.setdefault((word1, "<OOV>"), {})
                counts[(word1, "<OOV>")].setdefault("<OOV>", 0)

                counts[(word1, "<OOV>")]["<OOV>"] += 1

            # [(word1, oov)][word3] = 101
            elif (word1 in vocab) and (word2 not in vocab) and (word3 in vocab):
                counts.setdefault((word1, "<OOV>"), {})
                counts[(word1, "<OOV>")].setdefault(word3, 0)

                counts[(word1, "<OOV>")][word3] += 1

            # [(word1, word2)][oov] = 110
            elif (word1 in vocab) and (word2 in vocab) and (word3 not in vocab):
                counts.setdefault((word1, word2), {})
                counts[(word1, word2)].setdefault("<OOV>", 0)

                counts[(word1, word2)]["<OOV>"] += 1

            # [(word1, word2)][word3] - all in vocab = 111
            else:
                counts.setdefault((word1, word2), {})
                counts[(word1, word2)].setdefault(word3, 0)

                counts[(word1, word2)][word3] += 1

    return counts

def calculateProbs(unigramCts, bigramCts, trigramCts, wordMinus1 , wordMinus2 = None):

    vocabSize = len(unigramCts)
    bigramProbs = {}

    # set count to 1 for smoothing
    if wordMinus1 not in bigramCts:
        bigramCts[wordMinus1] = {}

    for key in bigramCts[wordMinus1].keys():
        # (c(i-1, i) + 1) / (c(i-1) + V)
        bigramProbs[key] = (bigramCts[wordMinus1][key] + 1) / (unigramCts[wordMinus1] + vocabSize) 

     # oov not originally in the bigram counts so we have to manually add it with one-smoothing.
    if "<OOV>" not in bigramProbs:
        bigramProbs["<OOV>"] =  (0 + 1) / (unigramCts[wordMinus1] + vocabSize)

    # We need to create trigram probs since we're given 2 words.
    trigramProbs = {}


    if wordMinus2 != None:
        if wordMinus2 not in bigramCts:
            bigramCts[wordMinus2][wordMinus1] = 1

        if (wordMinus2, wordMinus1) not in trigramCts:
            trigramCts[(wordMinus2, wordMinus1)] = {}
        
        for key in trigramCts[(wordMinus2, wordMinus1)].keys():
            # tri_prob = P(i given i-1 and i-2)
            tri_prob = (trigramCts[(wordMinus2, wordMinus1)][key] + 1) / (bigramCts[wordMinus2][wordMinus1] + vocabSize)

            # Interpolate the probability
            trigramProbs[key] = (bigramProbs[key] + tri_prob) / 2

        if "<OOV>" not in trigramProbs:
            # trigramCts[(wordMinus1, wordMinus2)]["<OOV>"] is 0 hence why it was never initially added to the probs.
            try:
                tri_oov_prob = (0 + 1) / (bigramCts[wordMinus2][wordMinus1] + vocabSize)
            except KeyError:
                tri_oov_prob = (0 + 1) / (1 + vocabSize)
            trigramProbs["<OOV>"] = (bigramProbs["<OOV>"] + tri_oov_prob) / 2

        return trigramProbs
    
    # else the bigram probs will suffice.
    else:
        return bigramProbs

# guaranteed at least one token ie len(words.split(" ")) >= 1.
def generateLanguage(words, unigramCts, bigramCts, trigramCts):

    output = []
    max_len = 32
    if len(words.split(" ")) == 1: # only given <s> to determine the next word from.
        minus1_probs = calculateProbs(unigramCts, bigramCts, trigramCts, "<s>")
        minus2 = "<s>"
        output.append(minus2)
        
    else: # given more words so we don't start with <s>
        output.extend(words.split(" "))
        minus1_probs = calculateProbs(unigramCts, bigramCts, trigramCts, output[len(output) - 1]) # last word of given text is our first word.
        minus2 = output[len(output) - 1]

    # renormalize probabilities
    prob_sum = sum(minus1_probs.values())
    for key in minus1_probs.keys():
        minus1_probs[key] /= prob_sum

    minus1 = np.random.choice([tup[0] for tup in minus1_probs.items()], p = [tup[1] for tup in minus1_probs.items()])
    output.append(minus1)
    while (len(output) != 32):
        word3_probs = calculateProbs(unigramCts, bigramCts, trigramCts, minus1, minus2)
        if not word3_probs:
            word3.append("</s>")
            break

        # renormalize probabilities
        prob_sum = sum(word3_probs.values())

        for key in word3_probs.keys():
            word3_probs[key] /= prob_sum

        word3 = np.random.choice([tup[0] for tup in word3_probs.items()], p = [tup[1] for tup in word3_probs.items()])

        output.append(word3)
        if word3 == "</s>":
            break
        
        minus2 = minus1 # previous i-1 is now i-2 for the next iteration.
        minus1 = word3 # new i-1 for the next iteration is the word we just created
    
    return " ".join(output)

def main():

    data = readData(sys.argv[1]) # use to train

    debug = False
    if debug:
        for part in data:
            print("{0}: ".format(part))
            print(data[part])
            print()
        sys.exit()
    # Create list of 5000 most frequent words.
    countsSorted = sorted(data["counts"].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    countsSorted = countsSorted[:5000] # we only want the 5000 most frequent ones.
    
    vocab = [tup[0] for tup in countsSorted]

    unigram_counts = extractUnigramCounts(data["counts"], vocab)
    
    bigram_counts = extractBigramCounts(data["contexts"], vocab)
    #print(bigram_counts)
    trigram_counts = extractTrigramCounts(data["contexts"], vocab)
    #for key in trigram_counts.keys():
        #print(key)
        #print("\t" + str(trigram_counts[key]))
    
    print("CHECKPOINT 2.2 - counts")
    print("UNIGRAMS:")
    try:
        print("language: " + str(unigram_counts["language"]))
    except KeyError:
        print("language: 0")
    try:
        print("the:" + str(unigram_counts["the"]))
    except KeyError:
        print("the: 0")
    try:
        print("formal:" + str(unigram_counts["formal"]))
    except KeyError:
        print("formal: 0")
    print("\n")

    print("BIGRAMS:")
    try:
        print("(the, language): " + str(bigram_counts["the"]["language"]))
    except KeyError:
        print("(the, language): 0")
    try:
        print("(<OOV>, language): " + str(bigram_counts["<OOV>"]["language"]))
    except KeyError:
        print("(<OOV>, language): 0")
    try:
        print("(to, process): " + str(bigram_counts["to"]["process"]))
    except KeyError:
        print("(to, process): 0")
    print("\n")

    print("TRIGRAMS:")
    try:
        print("(specific, formal, languages): " + str(trigram_counts[("specific", "formal")]["languages"]))
    except KeyError:
        print("(specific, formal, languages): 0")
    try:
        print("(to, process, <OOV>): " + str(trigram_counts[("to", "process")]["<OOV>"]))
    except KeyError:
        print("(to, process, <OOV>): 0")
    try:
        print("(specific, formal, event): " + str(trigram_counts[("specific", "formal")]["event"]))
    except KeyError:
        print("(specific, formal, event): 0")
    print("\n")

    print("\n")
    print("CHECKPOINT 2.3 - Probs with addone")

    print("BIGRAMS:")
    print("(the, language): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "the")["language"]))
    print("(<OOV>, language): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "<OOV>")["language"]))
    print("(to, process): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "to")["process"]))
    print("\n")

    print("TRIGRAMS:")
    print("(specific, formal, languages): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "formal", "specific")["languages"]))
    print("(to, process, <OOV>): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "process", "to")["<OOV>"]))
    try:
        print("(specific, formal, event): " + str(calculateProbs(unigram_counts, bigram_counts, trigram_counts, "formal", "specific")["event"]))
    except KeyError:
        print("(specific, formal, event): Not valid Wi")
    print("\n")

    print("FINAL CHECKPOINT: GENERATED LANGUAGE")
    print("PROMPT: <s>")
    for i in range(3):
        print(generateLanguage("<s>", unigram_counts, bigram_counts, trigram_counts))
        print("\n")
    print("\n")

    print("PROMPT: <s> language is")
    for i in range(3):
        print(generateLanguage("<s> language is", unigram_counts, bigram_counts, trigram_counts))
        print("\n")
    print("\n")

    print("PROMPT: <s> machines")
    for i in range(3):
        print(generateLanguage("<s> machines", unigram_counts, bigram_counts, trigram_counts))
        print("\n")
    print("\n")

    print("PROMPT: <s> they want to process")
    for i in range(3):
        print(generateLanguage("<s> they want to process", unigram_counts, bigram_counts, trigram_counts))
        print("\n")
    print("\n")

if __name__ == "__main__":
    main()