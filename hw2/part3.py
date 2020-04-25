import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import string

# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.nn = tnn.GRU(input_size=50,
                           hidden_size=50,
                           bidirectional=True,
                           num_layers=3,
                           batch_first=True)
        self.fc1 = tnn.Linear(50*2, 64)
        self.fc2 = tnn.Linear(64, 1)
        #self.fc3 = tnn.Linear(32, 1)
        #self.fc4 = tnn.Linear(16, 1)
        

    def forward(self, input, length):
        

        # encode from lstm network
        x_len_sorted, x_idx = torch.sort(length, descending=True)
        
        x_sorted = input.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)
        
        x_packed = tnn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, h = self.nn(x_packed)
        
        # resort h[0]
        h_size = h.size()
        h = h.permute(1, 0, 2).contiguous().view(-1, h_size[0] * h_size[2]).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)
        # (bidirection, batch size, embedding dim)
        h = h.view(-1, h_size[0], h_size[2]).permute(1, 0, 2).contiguous()
        
        h = torch.cat((h[-2], h[-1]), dim=1)

        # using h[0] as the seq representation, input into linear layer
        x = self.fc1(h)
        #x = self.drop(x)
        x = tnn.functional.relu(x)

        #x = self.fc2(x)
        #x = tnn.functional.relu(x)
        
        x = self.fc2(x).view(-1)
        return x

class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch
    stop_words = None#list(string.ascii_lowercase+string.punctuation)
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post, stop_words=stop_words)



def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    
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

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0
        total = 0
        num_correct = 0

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
            
            predicted = torch.round(torch.sigmoid(output.detach())).view(-1)
            num_correct += torch.sum(torch.eq(labels, predicted)).item()
            
            total += labels.size(0)
            
            loss = criterion(output, labels)
            
            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                #print("Train acc: %2f" % (num_correct/total))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length)).view(-1)
            predicted = torch.round(outputs).view(-1)
            
            num_correct += torch.sum(torch.eq(labels, predicted)).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
    net = Network()
    
    net.load_state_dict(torch.load('model.pth', map_location= lambda storage, loc: storage))
    
    torch.save(net.state_dict(), 'model_cpu.pth')
