import torch
import torch.nn as nn


def create_fake_data():
    """This function creates some fake data so we can test out our bespoke Convolutional Neural Net model architecture.
    The data we create splits the training data into two parts.
    One part of the data is the time series data representing an economic scenario with 3 values projected over 100 time periods.
    The other part of the data represents policyholder contract details that will be concatenated into the bespoke CNN model."""
    # Number of samples
    num_samples = 10
    # Length of each economic scenario projection
    sequence_length = 100
    # Number of channels, or indices we are projecting in the scenario
    num_channels = 3
    # Generate fake data with the shape (num_samples, num_channels, sequence_length)
    X_scenario = torch.randn(num_samples, num_channels, sequence_length)
    print("fake training data, X:", X_scenario)
    # generate extra policyholder features we concatenate into the CNN before the linear layer output
    X_extra = torch.randn(num_samples, 10)
    # Generate fake target values with shape (num_samples,) that range between 30 and 500
    y = torch.rand((num_samples,)) * 470 + 30
    # y = torch.randn(num_samples)
    print("fake targets, y:", y)
    # print("shape of the target in the training data:", y.shape)

    return X_scenario, X_extra, y


class ConvNet(nn.Module):
    """This code defines a 1D convolutional neural network class in PyTorch called ConvNet.
    The network has three Conv1D layers with increasing number of filters (32, 64, 128),
    each followed by a ReLU activation and average pooling.
    Before the final two linear layers, we concatenate in additional features that don't have a spatial relationship."""
    def __init__(self, extra_features_size):
        super(ConvNet, self).__init__()
        # Layer 1: Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        # Followed by a ReLU activation and average pooling with kernel size 2 and stride 2
        # the network can handle 3-channel input of shape (batch_size, 3, length).
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2))

        # Layer 2: Convolutional layer with 64 filters, kernel size 3, stride 1, padding 1
        # Followed by a ReLU activation and average pooling with kernel size 2 and stride 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2))

        # Layer 3: Convolutional layer with 128 filters, kernel size 3, stride 1, padding 1
        # Followed by a ReLU activation and average pooling with kernel size 2 and stride 2
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2))

        # Fully connected layer with not 128, but 1536 input units and 1 output unit
        # self.fc = nn.Linear(in_features=1536, out_features=1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1536 + extra_features_size, out_features=200),
            nn.ReLU())

        self.fc2 = nn.Linear(200, 1)

    def forward(self, scenario_x, policy_feature_x):
        """this method is called when we execute the model object function on an input"""
        # Pass input through the first layer
        out = self.layer1(scenario_x)

        # Pass the output of the first layer through the second layer
        out = self.layer2(out)

        # Pass the output of the second layer through the third layer
        out = self.layer3(out)

        # Flatten the output of the third layer to have 1536 units as input for the fully connected layer
        out = out.view(out.size(0), -1)
        # print("input for the fully-connected layer:", out.shape)

        # need to concatenate extra inputs into the fully connected layer before output
        # maybe add a ReLU to the linear layer and add one more fully connected layer?)
        out = torch.cat([out, policy_feature_x], dim=1)

        # Pass the output of the flatten layer with concatenated features through the fully connected layer with ReLu
        out = self.fc1(out)
        # Pass through final fully connected layer to 1 output
        out = self.fc2(out)

        # added this line to take out the nested singleton dimension to match the y values
        out = out.squeeze(-1)
        # print("cnn model output dimension:", out.shape)

        # Return the final output
        return out


def train_model(X, X_extra, y, cnn_model):
    """This code trains the cnn_model for 100 epochs using stochastic gradient descent (SGD) as the optimizer
    with a learning rate of 0.01 and momentum of 0.9. The mean squared error (MSE) loss function is used to
    evaluate the error between the model's predictions and the true target values.
    The loss is printed every epoch. Finally, the trained model's state is saved to a file called cnn_model.pth."""
    # Set the loss function
    criterion = nn.MSELoss()

    # Set the optimizer
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.0000001, momentum=0.9)

    epochs = 5000

    # Train the model
    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = cnn_model(X, X_extra)

        # Compute the loss
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Optimize the parameters
        optimizer.step()

        # Print the loss
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # Save the trained model
    # torch.save(cnn_model.state_dict(), 'cnn_model.pt')
    torch.save(cnn_model.state_dict(), '/Users/jonathankhanlian/Desktop/cnn_model.pth')
    return cnn_model


def predict(cnn_model, scenario_data, policy_data):
    """The input_data should have two parts: scenario data & policy data
    dimensions: batch_size x 3 channels x length.
    dimensions2: batch_size x 10 extra features
    The output of this function will be a batch_sized tensor containing the predicted values for each input.
    The outputs.squeeze function is used to remove the singleton dimension."""
    cnn_model.eval()
    with torch.no_grad():
        # input_tensor = torch.tensor(input_data, dtype=torch.float32)
        scenario_tensor = scenario_data.clone().detach()
        policy_tensor = policy_data.clone().detach()
        outputs = cnn_model(scenario_tensor, policy_tensor)
        prediction = outputs.squeeze()
    return prediction


# generate some fake data
X_scenario_data, X_policy_data, y = create_fake_data()
# construct the bespoke model architecture
cnn_model = ConvNet(10)  # extra features size is the argument
# train the model
trained_model = train_model(X_scenario_data, X_policy_data, y, cnn_model)
# predict on some new data (fake test data) using the trained model
predictions = predict(trained_model, torch.randn(5, 3, 100), torch.randn(5, 10))
print("predictions on five test records: 5 batch X 3 channels X 100 length + 5 batch X 10 length:", predictions)