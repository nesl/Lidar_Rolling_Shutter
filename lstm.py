import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from dataset_lstm import TimeSeriesDataset

class LSTMFreq(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(LSTMFreq, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*70, tagset_size)

    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(sentence)#.view(len(sentence),1,-1))
        tag_space = self.hidden2tag(lstm_out.view(len(lstm_out),-1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def testAccuracy(model, test_loader, NUM_SAMPLES, EMBEDDING_DIM):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            sentence, labels = data
            # run the model on the test set to predict labels
            outputs = model(sentence.view(-1, NUM_SAMPLES,EMBEDDING_DIM))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)
    

EMBEDDING_DIM = 14
HIDDEN_DIM = 28
NUM_EPOCHS = 50
tags = 9

params = {'batch_size': 40,
          'shuffle': True,
          'num_workers': 2}

data_dir = 'data'

training_set = TimeSeriesDataset(data_dir, 'train')
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = TimeSeriesDataset(data_dir, 'validation')
validation_generator = torch.utils.data.DataLoader(validation_set, **params)
        
model = LSTMFreq(EMBEDDING_DIM, HIDDEN_DIM, tags)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


for epoch in range(NUM_EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
    running_loss = 0.0
    for i, (sentence, tags) in enumerate(training_generator):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

  
        # Step 3. Run our forward pass.
        tag_scores = model(sentence.view(-1,training_set.NUM_SAMPLES,EMBEDDING_DIM))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()

        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()
        
        #loss_meter.update(loss.item(), params['batch_size'])
        running_loss += loss.item()     # extract the loss value
        if i % 10 == 9:    
            # print every 1000 (twice per epoch) 
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            # zero the loss
            running_loss = 0.0
        
    acc = testAccuracy(model, validation_generator, training_set.NUM_SAMPLES, EMBEDDING_DIM)
    
    print("Epoch", epoch, "Accuracy", acc)
"""
acc_meter.update(

print('Train [{}]/{}]\t{loss_meter}\t{acc_meter}'.format(epoch,     NUM_EPOCHS))
"""
        



