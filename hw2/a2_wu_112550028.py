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

def crossEntropyLoss(lemma, traindata, testdata):

    # for each context in each lemma, we wanna create 2 one-hots for every word.
    countsSorted = sorted(traindata["counts"], key=traindata["counts"].get, reverse=True) # sort (we lose the counts in the process but that's fine)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.

    countsSortedTest = sorted(testdata["counts"], key=testdata["counts"].get, reverse=True)
    countsSortedTest = countsSortedTest[:2000] # we only want the 2000 most frequent ones.

    xTrain = []
    yTrain = []
    for contextTup in traindata[lemma]: # loop through all the contexts for the current lemma
        before, after = createOneHots(contextTup, countsSorted)
        combined = before + after
        xTrain.append(combined)
        
        # append the corresponding sense that matches the current onehot for xtrain.
        for sense in traindata["senses"][lemma].keys():
            if traindata["senses"][lemma][sense] == contextTup[2]:
                yTrain.append(contextTup[2])
                break
        
    xTest = []
    yTest = []
    for contextTup in testdata[lemma]:
        before, after = createOneHots(contextTup, countsSortedTest)
        combined = before + after
        xTest.append(combined)

        for sense in testdata["senses"][lemma].keys():
            if testdata["senses"][lemma][sense] == contextTup[2]:
                yTest.append(contextTup[2])
                break
                
    # store in np array
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    xTest = np.array(xTest)
    yTest = np.array(yTest)

    # convert to tensor
    xTrain = torch.from_numpy(xTrain.astype(np.float32))
    yTrain = torch.from_numpy(yTrain.astype(np.int64)) # whole numbers

    xTest = torch.from_numpy(xTest.astype(np.float32))
    yTest = torch.from_numpy(yTest.astype(np.float32))

    '''
    xTrain = torch.tensor(xTrain)
    yTrain = torch.tensor(yTrain)

    xTest = torch.tensor(xTest)
    yTest = torch.tensor(yTest)
    '''

    # cross entropy loss
    learning_rate, epochs = 1.5, 400
    model = LogReg(list(xTrain.size())[1],6)
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
        #ytestpred_class = ytestpred_prob.round().type(torch.int8).numpy().T[0]
        #ytestpred_prob = ytestpred_prob.numpy().T[0]
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

def main():

    traindata = readData(sys.argv[1]) # use to train (x)
    testdata = readData(sys.argv[2]) # use to compare (y)
    # debugging
    debug = False
    if debug:
        for part in traindata:
            print("{0}: ".format(part))
            print(traindata[part])
            print()

    # part 1
    for lemma in traindata["lemmas"]:  # loop through all our lemmas
       #crossEntropyLoss(lemma, traindata, testdata) # TODO - **************** COMMENT BACK ****************************************************************
       pass
    
    # part 2
    countsSorted = sorted(traindata["counts"], key=traindata["counts"].get, reverse=True) # sort (we lose the counts in the process but that's fine)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.

    matrix = []
    # account for vocab smaller than 2000
    size = len(countsSorted)+1
    for i in range(size):
        matrix.append([0]*size)
    
    for lemma in traindata["lemmas"]: # loop through each lemma
        for contextTup in traindata[lemma]: # loop through contexts
            split = contextTup[1].split(" ")
            for i in range(len(split)): # loop through words in a given context.
                w1 = split[i] # row
                for j in range(i, len(split)):
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

                    # make sure to not increment twice for these examples.
                    # [OOA][OOA]
                    if indexw1 == size - 1 and indexw2 == size - 1:
                        matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                        continue
                    # diagonals
                    elif indexw1 == indexw2:
                        matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                        continue
                    matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                    matrix[indexw2][indexw1] = matrix[indexw2][indexw1] + 1

    #for row in matrix:
        #print(row)
    matrix = torch.tensor(matrix, dtype=torch.double)
    print(matrix[:,size-2])
    print(matrix.mean(dim=1, keepdim=True))
    print()
    print(matrix.std(dim=1, keepdim=True))
    test = matrix - matrix.mean(dim=1, keepdim=True)
    test2 = matrix / matrix.std(dim=1, keepdim=True, unbiased=False)
    matrix = (matrix - matrix.mean(dim=1, keepdim=True)) / matrix.std(dim=1, keepdim=True)
    svd = torch.svd(matrix)

if __name__ == '__main__':
    main()