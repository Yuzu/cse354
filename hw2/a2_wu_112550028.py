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
            finalcontext = ''
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

            data[lemma].append((headIndex, finalcontext, sense, lineID))

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

class LogReg(nn.Module):
    def __init__(self, num_feats, num_labels, learn_rate, device = torch.device("cpu") ):
        #DONT EDIT
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, num_labels) #add 1 to features for intercept

    def forward(self, X):
        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return self.linear(newX) #logistic function on the linear output

def runModel(lemma, traindata, testdata, xTrain, yTrain, xTest, yTest):
    learning_rate, epochs = 1.5, 650
    model = LogReg(list(xTrain.size())[1],6, learning_rate) # xTrain.size()[1] should be 4000, with 6 classes.
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    for i in range(epochs):
        model.train()
        sgd.zero_grad()
        #forward pass

        #torch.set_printoptions(profile="full")
        #print(yTrain)
        #return
        #sys.exit()
        #print(xTrain[32] == xTrain[92])
        ypred = model(xTrain)
        #print(ypred.shape) # [807,6]
        #print(yTrain.min()) # 0
        #print(yTrain.max()) # 5
        lossVal = loss(ypred, yTrain)
        # backwards
        lossVal.backward()
        sgd.step()

        #if i % 20 == 0:
            #print("  epoch %d, loss %.5f" %(i, lossVal.item()))
    
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
    countsSorted = sorted(traindata["counts"].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.
    countsSorted = [tup[0] for tup in countsSorted]

    countsSorted[1999] = "student" # TRY NOT TO DO THIS

    #print(countsSorted.index("student"))
    #print(countsSorted)
    #print (traindata["counts"]["student"])
    #sys.exit()
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

                    matrix[indexw1][indexw2] = matrix[indexw1][indexw2] + 1
                    matrix[indexw2][indexw1] = matrix[indexw2][indexw1] + 1

    #for row in matrix:
        #print(row)
    matrix = (matrix - np.mean(matrix, axis=1)) / np.std(matrix, axis=1)
    matrix = torch.tensor(matrix, dtype=torch.double)
    #print(matrix[:,size-2])
    #print(matrix.mean(dim=1, keepdim=True))
    #print()
    #print(matrix.std(dim=1, keepdim=True))
    u, s, v = torch.svd(matrix)
    embeddings = {}
    for i in range(size-1): # loop through all the words (rows) minus the OOV row.
        word = countsSorted[i]
        embeddings[word] = u[i][:50] # for each row take 50 columns

    embeddings["OOV"] = u[size-1][:50]
    # for embedding in embeddings.keys():
        #print(str(embedding) + ": " + str(embeddings[embedding]))

    
    language_process = torch.dist(embeddings["language"], embeddings["process"])
    print("('language', 'process') :", end="")
    print(language_process)

    machine_process = torch.dist(embeddings["machine"], embeddings["process"])
    print("('machine', 'process') :", end="")
    print(machine_process)
    
    language_speak = torch.dist(embeddings["language"], embeddings["speak"])
    print("('language', 'speak') :", end="")
    print(language_speak)

    # student isn't in the vocab.
    student_students = torch.dist(embeddings["student"], embeddings["students"])
    print("('student', 'students') :", end="")
    print(student_students)
    
    student_the = torch.dist(embeddings["student"], embeddings["the"])
    print("('student', 'the') :", end="")
    print(student_the)

    #print(torch.sum(embeddings["students"])) # tensor(-0.0010, dtype=torch.float64)
    #print(torch.sum(embeddings["the"])) # tensor(-0.7341, dtype=torch.float64)
    return embeddings

def main():

    traindata = readData(sys.argv[1]) # use to train
    testdata = readData(sys.argv[2]) # use to compare

    # THE INTEGER MAPPINGS FOR SENSES ARE DIFFERENT ACROSS THE TWO DATA SETS. WE WILL BE USING THE TRAINDATA'S MAPPINGS AS THE CORRECT ONES HERE:
    testdata["senses"] = traindata["senses"]

    #print(testdata["senses"])
    #print(traindata["senses"])
    #print(testdata["senses"] == traindata["senses"])

    # contextTup: [0] = index of target word, [1] = context, [2] = integer representation of sense, [3] = unique context ID
    # debugging
    debug = False
    if debug:
        for part in traindata:
            print("{0}: ".format(part))
            print(traindata[part])
            print()
    #sys.exit()

    #for word in traindata["counts"].keys():
        #total += traindata["counts"][word]
    #print(total)


    # Create list of 2000 most frequent words.
    countsSorted = sorted(traindata["counts"].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    countsSorted = countsSorted[:2000] # we only want the 2000 most frequent ones.
    countsSorted = [tup[0] for tup in countsSorted]

    countsSortedTest = sorted(testdata["counts"].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    countsSortedTest = countsSortedTest[:2000] # we only want the 2000 most frequent ones.
    countsSortedTest = [tup[0] for tup in countsSortedTest]

    '''
    # confirm consistency in one-hots
    first = traindata["process"][0] # "the process model"
    other = traindata["process"][5] # "the process of"
    fb, fa = createOneHots(first, countsSorted)
    ob, oa = createOneHots(other, countsSorted)
    print(fb == ob)
    print(fb.index(1))
    print(ob.index(1))
    return
    '''
    # part 1
    for lemma in traindata["lemmas"]:  # loop through all our lemmas
        #break # TODO - REMOVE HERE ****************************************
        xTrain = [] # 807 x 4000 tensors (1 for each context, each with 2 one-hots)
        yTrain = [] # 807 true values
        for contextTup in traindata[lemma]: # loop through all the contexts for the current lemma
            before, after = createOneHots(contextTup, countsSorted)
            combined = before + after
            xTrain.append(combined)

            # append the corresponding sense that matches the current onehot for xtrain.
            sense = contextTup[2]
            senseInt = traindata["senses"][lemma][sense]
            yTrain.append(senseInt)
        
        xTest = [] # 202 x 4000
        yTest = [] # 202
        for contextTup in testdata[lemma]:
            before, after = createOneHots(contextTup, countsSorted) # original = countssortedtest, LOOKS LIKE NEW ONE WORKS BETTER.
            combined = before + after
            xTest.append(combined)

            sense = contextTup[2]
            senseInt = testdata["senses"][lemma][sense]
            yTest.append(senseInt)

        '''
        # confirm the occurrence of 1s is betweeen [0, 2] - some words might not be in the vocab when creating the onehots
        for value in xTrain:
            if value.count(1) != 0 and value.count(1) != 1 and value.count(1) != 2:
                raise ValueError
        for value in xTest:
            if value.count(1) != 0 and value.count(1) != 1 and value.count(1) != 2:
                raise ValueError
        '''

        '''
        print(xTrain[32]) # "a process to" = process.NOUN.000041
        print(xTrain[91]) # "a process to" = process.NOUN.000115

        print(xTrain[32][:2000].index(1))
        print(xTrain[32][2000:].index(1))

        print(xTrain[91][:2000].index(1))
        print(xTrain[91][2000:].index(1))

        print(xTrain[32] == xTrain[91])
        print(countsSorted.index("a"))
        print(countsSorted.index("to"))

        sys.exit()
        '''
        xTrain = torch.tensor(xTrain, dtype=torch.float)
        yTrain = torch.tensor(yTrain, dtype=torch.long)

        xTest = torch.tensor(xTest, dtype=torch.float)
        yTest = torch.tensor(yTest, dtype=torch.long)

        '''
        for truelabel in yTrain:
            for sense in traindata["senses"][lemma].keys():
                if traindata["senses"][lemma][sense] == truelabel:
                    print(sense)
        sys.exit()
        '''

        '''
        torch.set_printoptions(profile="full")
        print(xTrain[32]) # "a process to" = process.NOUN.000041
        print(xTrain[91]) # "a process to" = process.NOUN.000115
        print(xTrain[32] == xTrain[91])
        sys.exit()
        '''
        runModel(lemma, traindata, testdata, xTrain, yTrain, xTest, yTest)
    
    # part 2
    #embeddings = extractWordEmbeddings(traindata)

    # part 3
    # For each context, we'll have a vector of length 200
    # by using the word embeddings we created (each len 50), we'll have [two words before] [one word before] _target_ [one word after] [two words after]
    # using this smaller data set as xTrain, we keep everything else the same.
if __name__ == '__main__':
    main()