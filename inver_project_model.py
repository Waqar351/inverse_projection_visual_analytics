import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

# Define the MLP inverse_model
class NNinv(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNinv, self).__init__()
        
        # Define the layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Input to first hidden layer
            nn.ReLU(),
            nn.Linear(64, 128),  # First hidden layer to second hidden layer
            nn.ReLU(),
            nn.Linear(128, 256),  # Second hidden layer to third hidden layer
            nn.ReLU(),
            nn.Linear(256, 512),  # Third hidden layer to fourth hidden layer
            nn.ReLU(),
            nn.Linear(512, output_size),  # Fifth hidden layer to output
            nn.Sigmoid()  # Output layer with sigmoid activation
        )
    
    def forward(self, x):
        return self.layers(x)
    

def model_train(input_size =2, output_size = 3, batch_size =64, X_train = None,y_train = None,X_test = None, y_test = None ):
    
    t_X_train = torch.tensor(X_train)
    t_y_train = torch.tensor(y_train)
    dataset = TensorDataset(t_X_train, t_y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the inverse_model, loss function, and optimizer
    inverse_model = NNinv(input_size, output_size)
    loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)

    # Number of epochs to train
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = inverse_model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print the average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")

    # Save the trained model
    torch.save(inverse_model.state_dict(), 'inverse_model.pth')

    # loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    # t_X_test = torch.tensor(X_test)
    # t_y_test = torch.tensor(y_test)
    # outputs_test = inverse_model(t_X_test)
    # loss_test = loss_fn(outputs_test, t_y_test)
    # print(loss_test/y_test.shape[0])

def model_test(input_size =2, output_size = 3, X_test = None, y_test = None, model = None):

    
    model.eval()  # Set the model to evaluation mode

    loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    t_X_test = torch.tensor(X_test)
    t_y_test = torch.tensor(y_test)
    outputs_test = model(t_X_test)
    loss_test = loss_fn(outputs_test, t_y_test)
    print(loss_test/y_test.shape[0])
