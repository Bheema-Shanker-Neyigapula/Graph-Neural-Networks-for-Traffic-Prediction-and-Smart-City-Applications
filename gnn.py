import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step A: Formulation of the Traffic Prediction Problem
# No code needed here as this is just a problem description

# Step B: Data Preprocessing and Graph Construction
# Load and preprocess the METR-LA dataset
def load_metr_la_data(data_path):
    traffic_data = pd.read_csv(data_path)

    # Drop any rows with missing values
    traffic_data.dropna(inplace=True)

    # Convert non-numeric values to NaN and then replace with column means
    traffic_data = traffic_data.apply(pd.to_numeric, errors='coerce')
    traffic_data.fillna(traffic_data.mean(), inplace=True)

    # Extract traffic flow values and scale them using StandardScaler
    scaler = StandardScaler()
    traffic_data_values = traffic_data.iloc[:, 1:].values  # Exclude the first column (timestamps)
    scaled_data_values = scaler.fit_transform(traffic_data_values)

    traffic_data.iloc[:, 1:] = scaled_data_values

    traffic_data = torch.tensor(traffic_data.values, dtype=torch.float32)

    return traffic_data

data_dir = 'METR-LA_data'
data_file_name = 'Metro_Interstate_Traffic_Volume.csv'

# Use the actual file path for the CSV file
data_path = os.path.join(data_dir, data_file_name)

# Add print statement to check the data path
print(f"Using data path: {data_path}")

traffic_data = load_metr_la_data(data_path)

# Define the prediction horizon
prediction_horizon = 12

# Prepare input and target data for the GNN model
num_timesteps = traffic_data.shape[0]
num_nodes = traffic_data.shape[1]
x, y = [], []

for i in range(num_timesteps - prediction_horizon):
    x.append(traffic_data[i:i + prediction_horizon])  # Historical traffic data (input)
    y.append(traffic_data[i + prediction_horizon])    # Target traffic data (to predict)

x = torch.stack(x)
y = torch.stack(y)

# Create graph data using PyTorch Geometric
edges = []
for i in range(num_nodes):
    for j in range(num_nodes):
        edges.append((i, j))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index, y=y)

# Split the data into training, validation, and testing sets
train_data = data[:800]
val_data = data[800:900]
test_data = data[900:]

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Step C: Designing the GNN Architecture
class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x, edge_index):
        x = x.permute(0, 2, 1)  # Reshape: (batch_size, num_nodes, num_timesteps) to (batch_size, num_timesteps, num_nodes)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return x

# Step D: Training and Fine-tuning the GNN Model
def train(model, train_loader, val_loader, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index)
                val_loss += criterion(output, data.y).item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}')

# Step D (continued): Testing the GNN Model
def test(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index)
            test_loss += criterion(output, data.y).item()

    print(f'Test Loss: {test_loss}')

# Step D (continued): Running the Code
if __name__ == "__main__":
    # Step C: Designing the GNN Architecture
    input_channels = prediction_horizon  # Number of input channels (traffic flow history for each node)
    hidden_channels = 16  # Number of hidden channels in the GCN layer
    output_channels = num_nodes  # Number of output channels (predicted traffic flow for each node)
    gnn_model = GraphConvolutionalNetwork(input_channels, hidden_channels, output_channels)

    # Step D: Training and Fine-tuning the GNN Model
    num_epochs = 50  # Specify the number of epochs for training
    learning_rate = 0.01  # Specify the learning rate
    train(gnn_model, train_loader, val_loader, num_epochs, learning_rate)

    # Step D (continued): Testing the GNN Model
    test(gnn_model, test_loader)
