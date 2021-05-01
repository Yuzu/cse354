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
    
    def __init__(self, inputLen, embeddingDimensions, hiddenStateDimensions, numTags):
        super(GRU_RNN, self).__init__()

        self.inputLen = inputLen
        self.embeddingDimensions = embeddingDimensions
        self.hiddenStateDimensions = hiddenStateDimensions
        self.numTags = numTags

        
        self.softmax = nn.LogSoftmax(dim=1)
        self.embedding = nn.Embedding(inputLen, hiddenStateDimensions)
        self.lstm = nn.LSTM(embeddingDimensions, hiddenStateDimensions)
        self.classifier = nn.Linear(hiddenStateDimensions, numTags)
    
    def forward(self, X, hiddenState):
        h0 = 0
        for i in range(1, len(X)):
            h1 = nn.GRU(len(X), self.hiddenStateDimensions, self.embeddingDimensions)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return newX, hiddenState #logistic function on the linear output
    
class LogReg(nn.Module):
    def __init__(self, num_feats, num_labels, learn_rate, device = torch.device("cpu") ):
        #DONT EDIT
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, num_labels) #add 1 to features for intercept

    def forward(self, X):
        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return self.linear(newX) #logistic function on the linear output

def runModel(xTrain, yTrain, xTest, yTest, learning_rate = 1.5, epochs = 650, penalty=0):
    model = LogReg(list(xTrain.size())[1],6, learning_rate) # xTrain.size()[1] should be 4000, with 6 classes.
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=penalty)
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
        
def main():

    # 1.1
    trainData = extractData("music_QA_train.json")
    #trialData = extractData("music_QA_dev.json")

    # 1.2
    word_embs = api.load('glove-wiki-gigaword-50')
    #print(word_embs['music'])
    #print(word_embs['unk'])

    for record in trainData:
        record['question_toks'] = tokenize(record['question'], lowercase=True)
        record['passage_toks'] = tokenize(record['passage'], lowercase=True)

    tokenEmbeddings = [] # len = amount of questions (419), of which are [len(passage), 50] size tensors
    trueLabels = [] # tensor of size len = amt of questions (419) by 1, has t/f values.
    for record in trainData:
        inputData = [get_embed(word, word_embs) for word in list(record['passage_toks']) + list(record['question_toks'])]
        '''
        print(len(list(record['passage_toks'])))
        print(list(record['passage_toks']))

        print("\n")
        print(len(list(record['question_toks'])))
        print(list(record['question_toks']))
        '''
        tokenEmbeddings.append(torch.tensor(inputData))
        trueLabels.append(record["label"])
    biggest = 0
    for tensor in tokenEmbeddings:
        if tensor.size()[0] > biggest:
            biggest = tensor.size()[0]
            
    tokenEmbeddings = pad_sequence(tokenEmbeddings)
    print(tokenEmbeddings.size())
    trueLabels = torch.tensor(trueLabels)
    #1.3
    runModel(tokenEmbeddings, trueLabels, None, None)

    #1.4
    print(tokenEmbeddings[0])



    #2.1
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model =  AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
if __name__ == "__main__":
    main()