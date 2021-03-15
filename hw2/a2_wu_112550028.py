# Tim Wu SBUID: 112550028
# CSE354, Spring 2021
# Assignment 2

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
            lineID = curline[0]


            curcontext = ''
            curcontext = curcontext.join(curline[2:]) # Anything beyond index 2 is the context.

            headIndex, curcontext = getTargetWord(curcontext)
            head = curcontext.split(" ")[headIndex].split("/")[0] # properly determine the index of the head after punctuation is removed.

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
            i = 0
            headfound = False
            for word in curcontext.split(" "):
                if word.split("/")[0] == head:
                    headIndex = i
                    headfound = True
                if word.split("/")[2] != "PUNCT":
                    finalcontext = finalcontext + " "

                # ignore punctuation
                if word.split("/")[2] == "PUNCT":
                    continue
                finalcontext += word.split("/")[0]
                if headfound is False:
                    i += 1

            finalcontext = finalcontext.strip()
            
            # check lemma to properly classify as a tuple of (headIndex, finalContext).
            lemma = curline[0].split(".")[0]

            # Assign an integer to this sense if we haven't seen it yet in the current lemma.
            sense = curline[1]
            if lemma not in data["senses"].keys():
                data["senses"][lemma] = {}

            if sense not in data["senses"][lemma].keys():
                data["senses"][lemma][sense] = sensecounter % 6
                sensecounter += 1

            # if a list for this lemma does not exist, we need to make it.
            if lemma not in data.keys():
                data[lemma] = [] # create list for lemma
                data["lemmas"].append(lemma) # also add it to the list of lemmas.

            data[lemma].append((headIndex, finalcontext, data["senses"][lemma][sense], lineID))

            
        return data

def createOneHots(contextTup: tuple, vocab):

    index = contextTup[0]
    context = contextTup[1]
    context = context.split(" ")
    before = [0] * len(vocab)
    if index != 0:
        word = context[index - 1]
        try:
            vocabindex = vocab.index(word)
            before[vocabindex] = 1
        except ValueError: # word not in the vocab, so just check after.
            pass

    after = [0] * len(vocab)
    if index != len(context) - 1:
        word = context[index + 1]
        try:
            vocabindex = vocab.index(word)
            after[vocabindex] = 1
        except ValueError:
            pass
    
    return before, after

## The Logistic Regression Class (do not edit but worth studying)
class LogReg(nn.Module):
    def __init__(self, num_feats, num_labels, learn_rate = 0.01, device = torch.device("cpu") ):
        #DONT EDIT
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, num_labels) #add 1 to features for intercept

    def forward(self, X):
        #DONT EDIT
        #This is where the model itself is defined.
        #For logistic regression the model takes in X and returns
        #a probability (a value between 0 and 1)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return self.linear(newX) #logistic function on the linear output

def crossEntropyLoss(lemma, traindata, testdata, xTrain, yTrain, xTest, yTest):
    # cross entropy loss
    learning_rate, epochs = 1, 650
    model = LogReg(list(xTrain.size())[1],6) # xTrain.size()[1] should be 4000, with 6 classes.
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    for i in range(epochs):
        model.train()
        sgd.zero_grad()
        #forward pass
        ypred = model(xTrain)
        #print(ypred.shape)
        #print(yTrain.min())
        #print(yTrain.max())
        lossVal = loss(ypred, yTrain)
        # backwards
        lossVal.backward()
        sgd.step()

        if i % 20 == 0:
            print("  epoch %d, loss %.5f" %(i, lossVal.item()))
    
    print("\nPredictions for {0}:".format(lemma))

    with torch.no_grad():
        ytestpred_prob = model(xTest)
        preds, inds = torch.max(ytestpred_prob, 1) # use this over the below approach since we want the highest in each tensor (which is the prediction)
        correct = 0
        for i in range(yTest.shape[0]):
            #print("  rating: %d,   logreg pred: %d" % (yTest[i], inds[i]))
            if yTest[i] == inds[i]:
                correct += 1
        print("Correct: {0}/{1}".format(correct, yTest.shape[0]))

    if lemma == "machine":

        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "machine.NOUN.000004":
                print("machine.NOUN.000004 " + str(ytestpred_prob[i].data))
                break

        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "machine.NOUN.000008":
                print("machine.NOUN.000008 " + str(ytestpred_prob[i].data))
                break
            
    elif lemma == "process":
        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "process.NOUN.000018":
                print("process.NOUN.000018 " + str(ytestpred_prob[i].data))
                break

        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "process.NOUN.000024":
                print("process.NOUN.000024 " + str(ytestpred_prob[i].data))
                break

    elif lemma == "language":

        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "language.NOUN.000008":
                print("language.NOUN.000008 " + str(ytestpred_prob[i].data))
                break

        for i in range(len(testdata[lemma])):
            if testdata[lemma][i][3] == "language.NOUN.000014":
                print("language.NOUN.000014 " + str(ytestpred_prob[i].data))
                break

def extractWordEmbeddings(traindata):
    countsSorted = sorted(traindata["counts"], key=traindata["counts"].get, reverse=True) # sort (we lose the counts in the process but that's fine)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.

    matrix = []
    size = len(countsSorted)+1
    for i in range(size):
        matrix.append([0]*size)
    
    for lemma in traindata["lemmas"]: # loop through each lemma
        for contextTup in traindata[lemma]: # loop through contexts
            split = contextTup[1].split(" ")
            for i in range(len(split)): # loop through words in a given context.
                w1 = split[i] # row
                for j in range(i+1, len(split)):
                    w2 = split[j] # col

                    # The words are stored with their original capitalization in the data array, but in the frequencies,
                    # they're stored as lowercase. to properly compare, we need to lowercase them.
                    w1 = w1.lower()
                    w2 = w2.lower()

                    try:
                        indexw1 = countsSorted.index(w1)
                    except ValueError:
                        indexw1 = size - 1
                    
                    try:
                        indexw2 = countsSorted.index(w2)
                    except ValueError:
                        indexw2 = size - 1

                    # make sure to not increment twice here.
                    # diagonals for co-occurrences
                    if indexw1 == indexw2:
                        matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                        continue

                    matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                    matrix[indexw2][indexw1] = matrix[indexw2][indexw1] + 1

    #for row in matrix:
        #print(row)
    matrix = torch.tensor(matrix, dtype=torch.double)
    #print(matrix[:,size-2])
    #print(matrix.mean(dim=1, keepdim=True))
    #print()
    #print(matrix.std(dim=1, keepdim=True))
    matrix = (matrix - matrix.mean(dim=1, keepdim=True)) / matrix.std(dim=1, keepdim=True)
    u, s, v = torch.svd(matrix)
    embeddings = {}
    for i in range(size-1): # loop through all the words (rows) minus the OOV row.
        word = countsSorted[i]
        embeddings[word] = u[i][:50] # for each row take 50 columns

    embeddings["OOV"] = u[size-1][:50]
    # for embedding in embeddings.keys():
        #print(str(embedding) + ": " + str(embeddings[embedding]))

    
    language_process = torch.dist(embeddings["language"], embeddings["process"])
    print(language_process)
    machine_process = torch.dist(embeddings["machine"], embeddings["process"])
    print(machine_process)
    language_speak = torch.dist(embeddings["language"], embeddings["speak"])
    print(language_speak)
    # student isn't in the vocab.
    student_students = torch.dist(embeddings["OOV"], embeddings["students"])
    print(student_students)
    student_the = torch.dist(embeddings["OOV"], embeddings["the"])
    print(student_the)

def main():

    traindata = readData(sys.argv[1]) # use to train
    testdata = readData(sys.argv[2]) # use to compare
    # contextTup: [0] = index of target word, [1] = context, [2] = integer representation of sense, [3] = unique context ID
    # debugging
    debug = False
    if debug:
        for part in traindata:
            print("{0}: ".format(part))
            print(traindata[part])
            print()

    # Create list of 2000 most frequent words.
    countsSorted = sorted(traindata["counts"], key=traindata["counts"].get, reverse=True) # sort (we lose the counts in the process but that's fine)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.

    countsSortedTest = sorted(testdata["counts"], key=testdata["counts"].get, reverse=True)
    countsSortedTest = countsSortedTest[:2000] # we only want the 2000 most frequent ones.
    '''
    first = traindata["process"][0]
    other = traindata["process"][5]
    fb, fa = createOneHots(first, countsSorted)
    ob, oa = createOneHots(other, countsSorted)
    print(fb == ob)
    print(fb.index(1))
    print(ob.index(1))
    '''
 
    # part 1
    for lemma in traindata["lemmas"]:  # loop through all our lemmas

        xTrain = [] # 807 x 4000 tensors (1 for each context, each with 2 one-hots)
        yTrain = [] # 807 true values
        for contextTup in traindata[lemma]: # loop through all the contexts for the current lemma
            before, after = createOneHots(contextTup, countsSorted)
            combined = before + after
            xTrain.append(combined)

            # append the corresponding sense that matches the current onehot for xtrain.
            yTrain.append(contextTup[2])
    
        xTest = [] # 202 x 4000
        yTest = [] # 202
        for contextTup in testdata[lemma]:
            before, after = createOneHots(contextTup, countsSortedTest)
            combined = before + after
            xTest.append(combined)

            yTest.append(contextTup[2])
        
        xTrain = torch.tensor(xTrain, dtype=torch.float)
        yTrain = torch.tensor(yTrain, dtype=torch.long)

        xTest = torch.tensor(xTest, dtype=torch.float)
        yTest = torch.tensor(yTest, dtype=torch.long)
        
        crossEntropyLoss(lemma, traindata, testdata, xTrain, yTrain, xTest, yTest)
    
    # part 2
    #extractWordEmbeddings(traindata)

    # part 3
    # For each context, we'll have a vector of length 200
    # by using the word embeddings we created (each len 50), we'll have [two words before] [one word before] _target_ [one word after] [two words after]
    # using this smaller data set as xTrain, we keep everything else the same.
if __name__ == '__main__':
    main()