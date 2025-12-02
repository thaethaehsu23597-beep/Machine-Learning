import torch
import torch.nn as nn
import torch.optim as optim

# xor dataset (classification)
x = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=torch.float32)

# class labels as integers
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

# Model


class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU()


# weight reused in forward pass


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


# create an instance of the network
model = XORNet()

# loss function and
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
epochs = 50
for epoch in range(epochs):

    optimizer.zero_grad()
# x is input
    outputs = model(x)

    loss = criterion(outputs, y)
    loss.backward()

    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# testing the model

test_inputs = torch.tensor([[1, 0],
                            [0, 1],
                            [1, 1],
                            [0, 0]])
test_inputs = test_inputs.float()
test_targets = torch.tensor([1, 1, 0, 0])

# with test model after training
with torch.no_grad():
    outputs = model(test_inputs)

    # apply softmax to get probabilities
    probs = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(probs, dim=1)

    print("Predicted labels:", predicted.tolist())
    print("True labels:", test_targets.tolist())
