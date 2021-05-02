# Tim Wu SBUID: 112550028
# CSE354, Spring 2021
# Assignment 4

import json
import sys
import gensim.downloader as api
from gensim.utils import tokenize
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# my system locale's character encoding is different, hence the need to explicitly mention utf-8.
sys.stdout = open('a4_wu_112550028.txt', 'w', encoding='utf-8')

def extractData(filename):
    data = []
    with open(filename, "r") as f:
        data = json.load(f)
    
    return data

def get_embed(word, embs):
    try:
        emb = embs[word]
    except KeyError:
        emb = embs['unk']
    
    return emb

class GRU_RNN(nn.Module):
    
    def __init__(self, inputSize, hiddenSize, numLabels = 2):
        super(GRU_RNN, self).__init__()

        self.lstm = nn.LSTM(inputSize, hiddenSize)
        self.linearClassifier = nn.Linear(hiddenSize, numLabels)
    
    def forward(self, X):
        hiddenStates = []
        for prompt in X:
            s, _ = self.lstm(prompt.unsqueeze(1))

            hiddenStates.append(s[-1])
        
        hiddenStates = torch.stack(hiddenStates).squeeze(1)

        hiddenStates = self.linearClassifier(hiddenStates)

        probs = nn.functional.softmax(hiddenStates, dim=1)
        return probs
    

def runModel(xTrain, yTrain, xTrial, yTrial, learning_rate = 1.5, epochs = 10, penalty=0):
    model = GRU_RNN(50, 50, 2)
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=penalty)
    loss = nn.BCELoss()

    for i in range(epochs):
        model.train()
        sgd.zero_grad()
        #forward pass

        ypred = model(xTrain)
        lossVal = loss(ypred, yTrain)

        # backwards
        lossVal.backward()
        sgd.step()
    
    print("PREDICTIONS:")
    with torch.no_grad():
        yTrial_probs = model(xTrial)
        # T/F
        correct = 0
        for i in range(yTrial.shape[0]):
            if yTrial[i][0] == 1:
                # True
                trueLabel = True
            else:
                trueLabel = False
            
            if yTrial_probs[i][0] > yTrial_probs[i][1]:
                # true
                predLabel = True
            else:
                predLabel = False
            
            if trueLabel == predLabel:
                correct += 1
        print("Correct: {0}/{1} = {2}".format(correct, yTrial.shape[0], correct/yTrial.shape[0]))

        
def main():

    # 1.1
    trainData = extractData("music_QA_train.json")
    trialData = extractData("music_QA_dev.json")

    # 1.2
    word_embs = api.load('glove-wiki-gigaword-50')
    #print(word_embs['music'])
    #print(word_embs['unk'])

    for record in trainData:
        record['question_toks'] = tokenize(record['question'], lowercase=True)
        record['passage_toks'] = tokenize(record['passage'], lowercase=True)

    for record in trialData:
        record['question_toks'] = tokenize(record['question'], lowercase=True)
        record['passage_toks'] = tokenize(record['passage'], lowercase=True)

    xTrain = [] # len = amount of questions (419), of which are [len(passage), 50] size tensors
    yTrain = [] # tensor of size len = amt of questions (419) by 1, has t/f values.
    for record in trainData:
        inputData = [get_embed(word, word_embs) for word in list(record['passage_toks']) + list(record['question_toks'])]
        '''
        print(len(list(record['passage_toks'])))
        print(list(record['passage_toks']))

        print("\n")
        print(len(list(record['question_toks'])))
        print(list(record['question_toks']))
        '''
        xTrain.append(torch.tensor(inputData))
        if record["label"] == True:
            yTrain.append(torch.tensor([1.0, 0.0]))
        else:
            yTrain.append(torch.tensor([0.0, 1.0]))
            
    #xTrain = pad_sequence(xTrain)
    yTrain = torch.stack(yTrain)

    xTest = []
    yTest = []
    for record in trialData:
        inputData = [get_embed(word, word_embs) for word in list(record['passage_toks']) + list(record['question_toks'])]

        xTest.append(torch.tensor(inputData))
        if record["label"] == True:
            yTest.append(torch.tensor([1.0, 0.0]))
        else:
            yTest.append(torch.tensor([0.0, 1.0]))

    yTest = torch.stack(yTest)

    #1.3
    runModel(xTrain, yTrain, xTest, yTest)

    #1.4



    #2.1
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model =  AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
if __name__ == "__main__":
    main()