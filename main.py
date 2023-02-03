import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt


def create_fake_data():
    """This function creates some fake data so we can test out our bespoke Convolutional Neural Net model architecture.
    The data we create splits the training data into two parts.
    One part of the data is the time series data representing an economic scenario with 3 values projected over 100 time periods.
    The other part of the data represents policyholder contract details that will be concatenated into the bespoke CNN model."""
    num_samples = 10
    sequence_length = 100  # Length of each economic scenario projection
    num_channels = 3  # Number of channels, or indices we are projecting in the scenario

    # Generate fake data with the shape (num_samples, num_channels, sequence_length)
    X_scenario = torch.randn(num_samples, num_channels, sequence_length)

    # generate extra policyholder features we concatenate into the CNN before the linear layer output
    X_extra = torch.randn(num_samples, 10)

    # Generate fake target values with shape (num_samples,) that range between 30 and 500
    y = torch.rand((num_samples,)) * 470 + 30

    return X_scenario, X_extra, y


def split_data(scenario_data, extra_features, target_y, split_percent):
    """split the data while making sure the two associated parts of the X records, and the y targets, stay connected"""
    train_idx_list = random.sample(range(len(target_y)), int(len(target_y) * split_percent))
    valid_idx_list = [i for i in range(len(target_y)) if i not in train_idx_list]

    X_train_pt1 = scenario_data[train_idx_list]
    X_train_pt2 = extra_features[train_idx_list]

    X_valid_pt1 = scenario_data[valid_idx_list]
    X_valid_pt2 = extra_features[valid_idx_list]

    y_train = target_y[train_idx_list]
    y_valid = target_y[valid_idx_list]

    return X_train_pt1, X_train_pt2, X_valid_pt1, X_valid_pt2, y_train, y_valid


class ConvNet(nn.Module):
    """This code defines a 1D convolutional neural network class in PyTorch called ConvNet.
    The network has three Conv1D layers with increasing number of filters (32, 64, 128),
    each followed by a ReLU activation and average pooling.
    Before the final two linear layers, we concatenate in additional features (that don't have a spatial/temporal relationship)."""
    def __init__(self, extra_features_size):
        super(ConvNet, self).__init__()
        # Layer 1: Convolutional layer with 32 filters, kernel size 3, stride 1, padding 1
        # Followed by a ReLU activation and average pooling with kernel size 2 and stride 2
        # the network can handle 3-channel input of shape (batch_size, 3, time series length).
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

        # Two fully connected layers, the first one having a ReLu
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1536 + extra_features_size, out_features=200),
            nn.ReLU())

        self.fc2 = nn.Linear(200, 1)

    def forward(self, scenario_x, policy_feature_x):
        """this method is called when we execute the model object function on an input"""
        # print("CNN input shape (other input will be incorporated after layer 3):", scenario_x.shape)

        # Pass input through the first layer
        out = self.layer1(scenario_x)
        # print("shape after layer 1:", out.shape)

        # Pass the output of the first layer through the second layer
        out = self.layer2(out)
        # print("shape after layer 2:", out.shape)

        # Pass the output of the second layer through the third layer
        out = self.layer3(out)
        # print("shape after layer 3:", out.shape)

        # Flatten the output of the third layer to have 1536 units as input for the fully connected layer
        out = out.view(out.size(0), -1)
        # print("input for the fully-connected layer:", out.shape)
        # print("shape after flattening for fully connected input:", out.shape)

        # need to concatenate extra inputs into the fully connected layer before output
        # maybe add a ReLU to the linear layer and add one more fully connected layer?)
        out = torch.cat([out, policy_feature_x], dim=1)
        # print("shape after concatenating additional features associated with the policy:", out.shape)

        # Pass the output of the flatten layer with concatenated features through the fully connected layer with ReLu
        out = self.fc1(out)
        # print("shape after first fully connected layer:", out.shape)
        # Pass through final fully connected layer to 1 output
        out = self.fc2(out)
        # print("shape after second fully connected layer:", out.shape)

        # added this line to take out the nested singleton dimension to match the y values
        out = out.squeeze(-1)
        # print("cnn model output dimension:", out.shape)
        # print("shape after removing nested singleton dimension:", out.shape)

        # Return the final output
        return out


def train_model(X, X_extra, CV_X, CV_X_extra, y, cv_y, cnn_model):
    """This code trains the cnn_model for 100 epochs using stochastic gradient descent (SGD) as the optimizer
    with a learning rate of 0.01 and momentum of 0.9. The mean squared error (MSE) loss function is used to
    evaluate the error between the model's predictions and the true target values.
    The loss is printed every epoch. Finally, the trained model's state is saved to a file called cnn_model.pth."""
    # Set the loss function
    loss_criterion = nn.MSELoss()

    loss_lists = []

    # Set the optimizer
    loss_optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.0000001, momentum=0.9)

    epochs = 5000

    # Train the model
    for epoch in range(epochs):
        # Zero the gradients, not the weights. We'll calc them again after forward and backward pass using the saved weights.
        loss_optimizer.zero_grad()

        # Forward pass
        outputs = cnn_model(X, X_extra)
        cv_outputs = cnn_model(CV_X, CV_X_extra)

        # Compute the loss
        loss = loss_criterion(outputs, y)
        cv_loss = loss_criterion(cv_outputs, cv_y)
        # log the loss for drawing loss curves
        # we need to detach() to avoid errors with Matplotlib. we don't need the hidden computational graph that lead to this loss anymore.
        loss_lists.append((loss.detach().numpy(), cv_loss.detach().numpy()))

        # Backward pass on the training data, not the CV data, to get the gradients
        loss.backward()

        # update the model weights by taking a gradient descent step down based on the gradients we just calculated
        loss_optimizer.step()

        # Print the loss
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, C-V_loss: {cv_loss}')

    # Save the trained model
    # torch.save(cnn_model.state_dict(), 'cnn_model.pt')
    torch.save(cnn_model.state_dict(), '/Users/jonathankhanlian/Desktop/cnn_model.pth')
    return cnn_model, loss_lists


def predict(cnn_model, scenario_data, policy_data):
    """The input_data should have two parts: scenario data & policy data
    dimensions: batch_size x 3 channels x length.
    dimensions2: batch_size x 10 extra features
    The output of this function will be a batch_sized tensor containing the predicted values for each input.
    The outputs.squeeze function is used to remove the singleton dimension."""
    cnn_model.eval()
    with torch.no_grad():
        scenario_tensor = scenario_data.clone().detach()
        policy_tensor = policy_data.clone().detach()
        outputs = cnn_model(scenario_tensor, policy_tensor)
        prediction = outputs.squeeze()
    return prediction


def generate_graphs(loss_curves):
    """this function will write loss curve images to png files"""
    train_loss_curve = [tup[0] for tup in loss_curves]
    cv_loss_curve = [tup[1] for tup in loss_curves]
    # print(type(train_loss_curve), type(train_loss_curve[0]), train_loss_curve[0])

    plt.bar([i for i in range(len(train_loss_curve))], train_loss_curve)
    plt.savefig("loss_curve_train.png")
    plt.clf()  # clear

    plt.bar([i for i in range(len(cv_loss_curve))], cv_loss_curve)
    plt.savefig("loss_curve_cross_val.png")
    plt.clf()  # clear


def run():
    """this does all our steps. generate data, split the data, train the model, generate loss curves, and test the trained model"""
    # generate some fake data
    X_scenario_data, X_policy_data, y = create_fake_data()

    # split the data
    X_train_pt1, X_train_pt2, X_valid_pt1, X_valid_pt2, y_train, y_valid = split_data(X_scenario_data, X_policy_data, y, 0.8)

    # construct the bespoke model architecture
    cnn_model = ConvNet(10)  # extra features size is the argument

    # train the model
    trained_model, loss_curve_lists = train_model(X_train_pt1, X_train_pt2, X_valid_pt1, X_valid_pt2, y_train, y_valid, cnn_model)

    # generate pngs of loss curves
    generate_graphs(loss_curve_lists)

    # predict on some new data (fake test data) using the trained model
    # we want the test data to be the same each time (although you may not) so we'll give it a seed
    random_seed = 42
    torch.manual_seed(random_seed)

    # 5 test cases, 100 period projection, 3 indices per period
    # 5 test cases, 10 policy features
    test_scenarios, test_policy_features = torch.randn(5, 3, 100), torch.randn(5, 10)

    predictions = predict(trained_model, test_scenarios, test_policy_features)
    print("predicted regression values on five economic scenarios, each with a unique set of policy features")
    print("input size: 5 records: 100 time periods X 3 channels/indices per period + 10 policy features:", predictions)


if __name__ == "__main__":
    print("run it")
    run()
