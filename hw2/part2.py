#!/usr/bin/env python3
"""
part2.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.

YOU MAY MODIFY THE LINE net = NetworkLstm().to(device)
"""

import numpy as np

import torch
import torch.nn as tnn
import torch.optim as topti

from torchtext import data
from torchtext.vocab import GloVe


# Class for creating the neural network.
class NetworkLstm(tnn.Module):
    """
    Implement an LSTM-based network that accepts batched 50-d
    vectorized inputs, with the following structure:
    LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkLstm, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.nn = tnn.LSTM(input_size=50,
                           hidden_size=100,
                           batch_first=True)
        self.fc1 = tnn.Linear(100, 64)
        self.fc2 = tnn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        # encode from lstm network
        
        x_len_sorted, x_idx = torch.sort(length, descending=True)
        
        x_sorted = input.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)
        
        x_packed = tnn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.nn(x_packed)
        
        # resort h[0]
        h_size = h.size()
        h = h.permute(1, 0, 2).contiguous().view(-1, h_size[0] * h_size[2]).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)
        # (bidirection, batch size, embedding dim)
        h = h.view(-1, h_size[0], h_size[2]).permute(1, 0, 2).contiguous()
        
        # using h as the seq representation, input into linear layer
        x = self.fc1(h[0])
        x = tnn.functional.relu(x)
        
        # the 2-th linear layer
        x = self.fc2(x).view(-1)
        return x


# Class for creating the neural network.
class NetworkCnn(tnn.Module):
    """
    Implement a Convolutional Neural Network.
    All conv layers should be of the form:
    conv1d(channels=50, kernel size=8, padding=5)

    Conv -> ReLu -> maxpool(size=4) -> Conv -> ReLu -> maxpool(size=4) ->
    Conv -> ReLu -> maxpool over time (global pooling) -> Linear(1)

    The max pool over time operation refers to taking the
    maximum val from the entire output channel. See Kim et. al. 2014:
    https://www.aclweb.org/anthology/D14-1181/
    Assume batch-first ordering.
    Output should be 1d tensor of shape [batch_size].
    """

    def __init__(self):
        super(NetworkCnn, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        # three cnn layers
        self.conv1 = tnn.Conv1d(50, 50, 8, padding=5)
        
        self.conv2 = tnn.Conv1d(50, 50, 8, padding=5)

        self.conv3 = tnn.Conv1d(50, 50, 8, padding=5)
        
        # the first pooling layer with size=4
        self.pool1 = tnn.MaxPool1d(4)
        
        # the last linear 
        self.fc1 = tnn.Linear(50, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        # permute input: the shape whill be convert into batch_size x seq_len x vec_dim from batch_size x vec_dim x seq_len 
        input = input.permute(0, 2, 1)
        
        # the 1-th cnn layer
        x = self.conv1(input)
        x = tnn.functional.relu(x)
        x = self.pool1(x)
        
        # the 2-th cnn layer
        x = self.conv2(x)
        x = tnn.functional.relu(x)
        x = self.pool1(x)
        
        # the 3-th cnn layer
        x = self.conv3(x)
        x = tnn.functional.relu(x)
        x = self.pool1(x)
        
        # max pooling over time
        x = tnn.functional.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        
        # input linear layer
        x = self.fc1(x).view(-1)
        
        return x
        
        
def lossFunc():
    """
    TODO:
    Return a loss function appropriate for the above networks that
    will add a sigmoid to the output and calculate the binary
    cross-entropy.
    """
    # define a function, inputs are "outputs" and "target"
    # "outputs" comes from the last linear layer of the model
    # "target" is the true target label 
    def loss_fn(outputs, trg):
        # calculate sigmoid value of outputs
        outputs = torch.sigmoid(outputs)
        # resize the shape of outputs
        outputs = outputs.view(trg.size(0))
        # calculate the binary cross entropy loss between outputs and target
        loss = tnn.functional.binary_cross_entropy(outputs, trg)
        
        return loss
    # return the loss function
    return loss_fn
    


def measures(outputs, labels):
    """
    TODO:
    Return (in the following order): the number of true positive
    classifications, true negatives, false positives and false
    negatives from the given batch outputs and provided labels.

    outputs and labels are torch tensors.
    """
    x = torch.sigmoid(outputs)
    
    x = x.view(labels.size(0))

    pred = torch.round(x)

    tp = ((torch.eq(pred, 1)) * (torch.eq(labels, 1))).sum().item()
    tn = ((torch.eq(pred, 0)) * (torch.eq(labels, 0))).sum().item()
    
    fp = ((torch.eq(pred, 1)) * (torch.eq(labels, 0))).sum().item()
    fn = ((torch.eq(pred, 0)) * (torch.eq(labels, 1))).sum().item()
    
    return tp, tn, fp, fn
    
    


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True)
    labelField = data.Field(sequential=False)

    from imdb_dataloader import IMDB
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    # Create an instance of the network in memory (potentially GPU memory). Can change to NetworkCnn during development.
    net = NetworkLstm().to(device)
    #net = NetworkCnn().to(device)

    criterion = lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()
