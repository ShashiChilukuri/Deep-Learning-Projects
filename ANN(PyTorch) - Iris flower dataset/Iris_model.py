import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):

        super(Network, self).__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in 
                                   zip(hidden_layers[:-1], hidden_layers[1:])])
        self.output = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):      
        for each in self.hidden_layers:
            x = F.relu(each(x))
        x = self.output(x)
        
        return x

def test(model, test_loader, criterion):
    test_accuracy= 0
    test_loss = 0
    
    for test_inputs, test_labels in test_loader:
        test_outputs = model(Variable(test_inputs))
        
        # Calculating Test_Loss
        test_loss += criterion(test_outputs, Variable(test_labels)).item()
        
        # Calculating test_accuracy
        _, predicted = torch.max(test_outputs.data, 1)
        test_correct = (predicted == test_labels)
        test_accuracy += 100 * test_correct.type_as(torch.FloatTensor()).mean()   
    
    return test_loss, test_accuracy


def train(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_loss = 0
    train_accuracy = 0

    for e in range(epochs):  
        for train_inputs, train_lables in train_loader:
        
            # Convert torch tensor to Variable
            train_inputs = Variable(train_inputs)
            train_lables = Variable(train_lables)
        
            model.train()                        # Enabling the network training mode
        
            optimizer.zero_grad()                # clearing previous gradients
            train_outputs = model(train_inputs)  # Forward propagation
            train_loss = criterion(train_outputs, train_lables) # Calculate the loss
            train_loss.backward()                # Back Propagation
            optimizer.step()                     # Adjusting the parameters
            
            # Calculating training loss
            train_loss += train_loss.item()
        
            # Calculating training accuracy
            _, predicted = torch.max(train_outputs.data, 1)
            train_correct = (predicted == train_lables)
            train_accuracy = 100 * train_correct.type_as(torch.FloatTensor()).mean()

        model.eval()                            # Enabling the network in evaluation mode 
        with torch.no_grad():                   # Turn off gradients for testing
                    test_loss, test_accuracy = test(model, test_loader, criterion)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Train Loss: {:.3f}.. ".format(train_loss),
              #"Train Accuracy: {:.3f}.. ".format(train_accuracy),
              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
              "Test Accuracy: {:.3f}".format(test_accuracy/len(test_loader)))
        
        model.train()                           # Make sure dropout and grads are on for training