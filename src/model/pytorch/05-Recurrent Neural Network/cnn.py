import torch 
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime


start = datetime.now()
loopstart=start

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

#MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)
test_dataset = torchvision.datasets.MNIST(root='../data',
                                         train=False,
                                         transform=transforms.ToTensor())

#Data loader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        #Forward progagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape(batch_size, seq_length, hidden_size)
        
        #Decode the hidden state of teh last time step
        out = self.fc(out[:, -1, :])
        return out
    
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labels.to(device)
        
        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            cur = datetime.now()
            print("Epoch [{}/{}], Step [{}/{}] Loss {:.4f} {}".format(epoch+1, num_epochs, i+1, total_step, loss.item(), cur-loopstart))
            loopstart = cur
            
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length,input_size).to(device)
        labels = labells.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
# Save the model checkpoint
torch.save(model.state_dict(), 'cnn.ckpt')
print("total time is {}".format(datetime.now() - start))