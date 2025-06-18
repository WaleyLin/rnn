import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

df = pd.read_csv("./coin_Bitcoin.csv")
x = df[["High", "Low", "Open"]].values
y = df[["Close"]].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

train_x = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)
seq_len = train_x[0].shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitCoinDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

hidden_size = 128
num_layers = 5
learning_rate = 0.001
batch_size = 64
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

rnn = RNN(input_feature_size=3, hidden_size=hidden_size, num_layers=num_layers).to(device)
criteria = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

rnn.train()
for epoch in range(epoch_size):
    loss_total = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = rnn(inputs)
        loss = criteria(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss_total / 100:.3f}')
            loss_total = 0.0

print('Finished Training')

prediction = []
ground_truth = []

rnn.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().cpu().flatten().tolist()
        prediction += out

prediction = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1))
ground_truth = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1))

r2score = r2_score(ground_truth, prediction)