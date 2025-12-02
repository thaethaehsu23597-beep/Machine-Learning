import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
# torch is the library
# generate data for training for four set of number


def generate_data(low, high, n_rows, n_cols):
    x_data = torch.randint(low, high, (n_rows, n_cols), dtype=torch.float32)
    y_data = torch.sum(x_data, dim=1, keepdim=True)
    return x_data, y_data


# define a simple neural network model
'''
nn.model = neural network of model 
defining the neural network fc1= fully connected layer
dense layer = fully connected layer
just defining the layer'''


class SimpleNN(nn.Module):
    def __init__(self, n_features):
        super(SimpleNN, self).__init__()
        # 1st hidden layer (100 neurons in 1st layers)
        # fc1 is the instance variable in ML model
        self.fc1 = nn.Linear(n_features, 100)
        # 2nd hidden layer (previous has 100 and 59 neurons )
        self.fc2 = nn.Linear(100, 50)
        # 3rd hidden layer (previous 50 and has 10 neurons)
        self.fc3 = nn.Linear(50, 10)  # 3rd hidden layer
        self.fc4 = nn.Linear(10, 1)  # 4th hidden layer
        self.relu = nn.ReLU()  # activation layer, create relu obj
    '''
    how the laywr are being arranged forward path
    fc1 is going to do the weighted sum
    final x is the predicted value of n1+ n2 + n3'''

    def forward(self, x):  # network architecture
        x = self.fc1(x)    # weighted sum for first layer
        x = self.relu(x)  # activation after each layer
        x = self.fc2(x)
        x = self.relu(x)  # activation after each layer
        x = self.fc3(x)
        x = self.relu(x)  # activation after each layer
        x = self.fc4(x)
        return x


# train the neural network x train is the training y train is the label
def train_model(model, criterion, optimizer, x_train, y_train, epochs):
    model.train()  # training model
    losses = []  # epochs 200 = loses 200

    for epoch in range(epochs):
        optimizer.zero_grad()  # to say whatever weight computed and then erase it
        # forward class calling model -> obj (x_train)
        outputs = model(x_train)
        loss = criterion(outputs, y_train)  # loss (MSE)
        # TO GET THE SINGLE VALUE -> RUNNING 500 ROWS ONE TIME GET ONLY ONE LOSS
        print("loss =", loss.item())
        '''
        loss will be smaller and smaller '''
        loss.backward()  # compute gradients from back to front
        optimizer.step()  # update weights based on computed gradients from back to front
        losses.append(loss.item())

    return losses


# test the neural network
def test_model(model, criterion, x_test, y_test):
    model.eval()  # set mode to evaluation mode
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test)
    return loss.item()


# show ground-truths vs predictions
def show_diffs(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predicts = model(x_test)

        # in case we have been using gpu
        output = torch.cat((x_test.cpu(), y_test.cpu(), predicts.cpu()), dim=1)

        df = pd.DataFrame(data=output.numpy(),
                          columns=["x1", "x2", "x3", "y", "predicted"])
        print(df)


# plot the loss curve
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Loss curve')
    plt.show()


# main program
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # cpu will be much slower

# prepare training and testing data
x_train, y_train = generate_data(-100, 100, 500, 3)
x_test, y_test = generate_data(-100, 100, 20, 3)

# use gpu if available
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

# create the neural network model
model = SimpleNN(n_features=x_train.shape[1]).to(device)
criterion = nn.MSELoss()
# oprimizer and criterion are objs
optimizer = optim.Adam(model.parameters(), lr=0.015)

# print model summary
print(model)

# train the model
print("Training the model...")
losses = train_model(model, criterion, optimizer, x_train, y_train, epochs=200)
plot_loss(losses)

# test the model
print("Testing the model...")
loss = test_model(model, criterion, x_test, y_test)
print(f'Loss = {loss}\n')

# show predictions vs actual values
show_diffs(model, x_test, y_test)
