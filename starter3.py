import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from Q1 import Q1_Net

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)
               
def classify_insurability():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    model = Q1_Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    n_epochs = 500
    batch_size = 2
    for epoch in range(n_epochs):
        for i in range(0, len(train), batch_size):
            X = torch.FloatTensor(train[i][1])
            label = train[i][0]
            y_target = torch.Tensor([0, 0, 0])
            y_target[label] = 1.0
            optimizer.zero_grad()
            y_pred = model.forward(X)
            loss = loss_fn(y_pred, y_target)
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
    
    test = np.array(test, dtype=object)
    test_features = test[:, 1]
    test_labels = test[:, 0]
    confusion_matrix = np.zeros((3, 3))
    count = 0
    for i in range(len(test_labels)):
        X = torch.FloatTensor(test_features[i])
        X = model.forward(X)
        y_pred = model.softmax(X)
        label = torch.argmax(y_pred)
        confusion_matrix[label][test_labels[i]] += 1
        if label.item() == test_labels[i][0]:
            count += 1
    print(f"The accuracy is {count/len(test_labels)}")
    print(confusion_matrix)
    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    
def classify_mnist_reg():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    
def classify_insurability_manual():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    classify_insurability()
    # classify_mnist()
    # classify_mnist_reg()
    # classify_insurability_manual()
    
if __name__ == "__main__":
    main()
