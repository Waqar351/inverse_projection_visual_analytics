import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

# ## Define the MLP inverse_model
# class NNinv(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(NNinv, self).__init__()
        
#         # Define the layers
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, 64),  # Input to first hidden layer
#             nn.ReLU(),
#             nn.Linear(64, 128),  # First hidden layer to second hidden layer
#             nn.ReLU(),
#             nn.Linear(128, 256),  # Second hidden layer to third hidden layer
#             nn.ReLU(),
#             nn.Linear(256, 512),  # Third hidden layer to fourth hidden layer
#             nn.ReLU(),
#             nn.Linear(512, output_size),  # Fifth hidden layer to output
#             nn.Sigmoid()  # Output layer with sigmoid activation
#         )
    
#     def forward(self, x):
#         return self.layers(x)

class NNinv(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNinv, self).__init__()
        
        # Define the layers with improvements
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),  # Input to first hidden layer
            nn.BatchNorm1d(128),         # Batch Normalization
            nn.LeakyReLU(0.1),           # Leaky ReLU activation
            nn.Dropout(0.2),             # Dropout for regularization
            
            nn.Linear(128, 256),         # First hidden layer to second hidden layer
            nn.BatchNorm1d(256),         # Batch Normalization
            nn.LeakyReLU(0.1),           # Leaky ReLU activation
            nn.Dropout(0.2),             # Dropout
            
            nn.Linear(256, 512),         # Second hidden layer to third hidden layer
            nn.BatchNorm1d(512),         # Batch Normalization
            nn.LeakyReLU(0.1),           # Leaky ReLU activation
            nn.Dropout(0.2),             # Dropout
            
            nn.Linear(512, output_size), # Output layer
            nn.Sigmoid()                 # Sigmoid activation for normalized output
        )
    
    def forward(self, x):
        return self.layers(x)

###--________ Testing purpose_________________

# import torch.nn.functional as F
# class NNinv(nn.Module):
#     def __init__(self, input_size, output_size, hidden_sizes=[128, 256, 512], dropout_rate=0.2):
#         super(NNinv, self).__init__()
        
#         self.hidden_layers = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()

#         # First layer
#         self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))

#         # Hidden layers with skip connections
#         for i in range(1, len(hidden_sizes)):
#             self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
#             self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i]))

#         # Output layer
#         self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

#         # Dropout rate
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         residual = x  # Store input for skip connections
#         for i, layer in enumerate(self.hidden_layers):
#             x = layer(x)
#             x = self.batch_norms[i](x)
#             x = F.leaky_relu(x, 0.1)
#             x = self.dropout(x)
            
#             # Skip Connection: Adding input features back (only if dimensions match)
#             if x.shape == residual.shape:
#                 x += residual
        
#         x = self.output_layer(x)
#         x = torch.tanh(x)  # Use Tanh for output activation
#         return x
    

    

# def model_train(epochs = 10,input_size =2, output_size = 3, batch_size =64, X_train = None,y_train = None, out_folder = None ):
    
#     t_X_train = torch.tensor(X_train)
#     t_y_train = torch.tensor(y_train)
#     dataset = TensorDataset(t_X_train, t_y_train)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Instantiate the inverse_model, loss function, and optimizer
#     inverse_model = NNinv(input_size, output_size)
#     loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
#     optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)

#     # Number of epochs to train
#     num_epochs = epochs

#     # Training loop
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (inputs, targets) in enumerate(dataloader):
#             # Forward pass
#             outputs = inverse_model(inputs)
#             loss = loss_fn(outputs, targets)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
        
#         # Print the average loss for the epoch
#         avg_loss = running_loss / len(dataloader)
#         # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

#     print("Training complete.")

#     # Save the trained model
#     torch.save(inverse_model.state_dict(), f'{out_folder}/inverse_model.pth')

#     return inverse_model

def model_train(epochs=10, input_size=2, output_size=3, batch_size=64, X_train=None, y_train=None, out_folder=None, patience=30, tolerance=1e-4):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim

    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(t_X_train, t_y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the inverse_model, loss function, and optimizer
    inverse_model = NNinv(input_size, output_size)
    loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)

    # Variables to track the best model and early stopping
    best_model_state = None
    lowest_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Forward pass
            outputs = inverse_model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Check for improvement in loss
        if avg_loss < lowest_loss - tolerance:
            lowest_loss = avg_loss
            best_model_state = inverse_model.state_dict()
            best_epoch = epoch
            patience_counter = 0  # Reset counter if improvement is observed
        else:
            patience_counter += 1
            # print(f"No significant improvement. Patience: {patience_counter}/{patience}")

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best Epoch {best_epoch}")

    # # Save the best model
    # if best_model_state is not None:
    #     torch.save(best_model_state, f'{out_folder}/best_inverse_model.pth')
    #     inverse_model.load_state_dict(best_model_state)
    
    # Save the best model only if out_folder is not None
    if out_folder and best_model_state is not None:
        torch.save(best_model_state, f'{out_folder}/best_inverse_model.pth')
        inverse_model.load_state_dict(best_model_state)
    else:
        print("Model not saved. Output folder is not specified.")

    return inverse_model


def model_test(X_test = None, y_test = None, model = None, bLossFlag = True):

    
    model.eval()  # Set the model to evaluation mode

    loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    t_X_test = torch.tensor(X_test)
    t_y_test = torch.tensor(y_test)
    outputs_test = model(t_X_test)

    if bLossFlag:
        loss_test = loss_fn(outputs_test, t_y_test)
        loss = loss_test/y_test.shape[0]
    else:
        loss = None
    return outputs_test, loss
