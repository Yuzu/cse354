# Tim Wu SBUID: 112550028
# CSE354, Spring 2021
# Assignment 4

import json
import sys
import gensim.downloader as api
from gensim.utils import tokenize
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def main():

    # 1.1
    trainData = extractData("music_QA_train.json")
    trialData = extractData("music_QA_dev.json")

    # 1.2
    go = False
    if go == True:
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

        #1.3/1.4
        #runModel(xTrain, yTrain, xTest, yTest)


    #2.1
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model =  AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    test = tokenizer("sentence one", "sentence two!")


    # mnli

    tokenizedTrain = []
    tokenizedTrial = []

    tokenizedTrain_q = []
    tokenizedTrain_p = []

    tokenizedTrial_q = []
    tokenizedTrial_p = []

    # Taken from Hugging Face SQuAD Tutorial (BEGIN)

    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    for record in trainData:
        entry = {}
        entry["input_ids"] =(tokenizer(record['question'], record['passage'],
            max_length=max_length, truncation="only_second", return_overflowing_tokens=True, stride=doc_stride))
        entry["idx"] = record["idx"]
        entry["label"] = record["label"]
        tokenizedTrain.append(entry)

    for record in trialData:
        entry = {}
        entry["input_ids"] = (tokenizer(record['question'], record['passage'],
            max_length=max_length, truncation="only_second", return_overflowing_tokens=True, stride=doc_stride))
        entry["idx"] = record["idx"]
        entry["label"] = record["label"]
        tokenizedTrial.append(entry)

    #for x in tokenizedTrain[5]["input_ids"][:2]:
        #print(tokenizer.decode(x))

    #sequence_ids = tokenizedTrain[5].sequence_ids()
    #print(sequence_ids)

    # (END)

    for sequence in tokenizedTrain:
        input_ids = sequence["input_ids"]
        q = {}
        p = {}

        q["idx"] = sequence["idx"]
        q["label"] = sequence["label"]
        p["idx"] = sequence["idx"]
        p["label"] = sequence["label"]

        q["input_ids"] = []
        p["input_ids"] = []

        for i, i_id in enumerate(input_ids.sequence_ids()):
            if i_id == None:
                continue
            elif i_id == 0:
                #print(tokenizer.decode(sequence["input_ids"][:2][0][i]))
                thing = input_ids["input_ids"][:2]
                q["input_ids"].append(input_ids["input_ids"][:2][0][i])
            else:
                #print(tokenizer.decode(sequence["input_ids"][:2][0][i]))
                p["input_ids"].append(input_ids["input_ids"][:2][0][i])
        
        tokenizedTrain_q.append(q)
        tokenizedTrain_p.append(p)

    args = TrainingArguments(
        "test-glue",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenizedTrain_q,
        eval_dataset=tokenizedTrain_q,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()

    print("a")

if __name__ == "__main__":
    main()