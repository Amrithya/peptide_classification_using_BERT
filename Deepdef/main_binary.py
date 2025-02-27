import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import random

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Character encoding
char_to_index = {ch: i+1 for i, ch in enumerate("ARNDBCEQZGHILKMFPSTWYVXOJU")}

file_path = "neoag_train.csv"
df = pd.read_csv(file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_seq(seq):
    return torch.tensor([char_to_index.get(char, 0) for char in seq], dtype=torch.long)

seqs = df["sequence"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
encoded_seqs = [encode_seq(seq) for seq in seqs]
labels = torch.tensor(labels, dtype=torch.float32)

def collate_fn(batch):
    seq_batch, label_batch = zip(*batch)
    seq_batch = pad_sequence(seq_batch, batch_first=True, padding_value=0)
    return seq_batch, torch.tensor(label_batch, dtype=torch.float32)

X_train, X_temp, y_train, y_temp = train_test_split(encoded_seqs, labels, test_size=0.2, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))
test_data = list(zip(X_test, y_test))

neg_count, pos_count = df["label"].value_counts()
pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)], dtype=torch.float32).to(device)

class seqClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate):
        super(seqClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.gru = nn.GRU(256, hidden_dim, batch_first=True, num_layers=num_layers, 
                          bidirectional=False, dropout=dropout_rate)
        
        self.fc = nn.Linear(hidden_dim, 1)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        
        x = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        
        x, _ = self.gru(x.permute(0, 2, 1))
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

def objective(trial):
    embed_dim = trial.suggest_int("embed_dim", 128, 512, step=64)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)

    model = seqClassifier(len(char_to_index) + 1, embed_dim, hidden_dim, num_layers, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    for epoch in range(5):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predicted.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds)
        trial.report(f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)

best_params = study.best_params
best_params_without_lr = {key: value for key, value in best_params.items() if key != 'learning_rate'}
model = seqClassifier(len(char_to_index) + 1, **best_params_without_lr).to(device)
learning_rate = best_params['learning_rate']
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

f1_scores = []
for epoch in range(50):
    model.train()
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(predicted.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    f1_scores.append(f1)
    print(f"Epoch {epoch+1}: F1-score = {f1:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), f1_scores, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("F1-score vs Epoch")
plt.grid()
plt.show()

best_epoch = np.argmax(f1_scores) + 1
best_f1 = max(f1_scores)
print(f"Best Epoch: {best_epoch}, Best F1-score: {best_f1:.4f}")

model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for seqs, labels in DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn):
        seqs, labels = seqs.to(device), labels.to(device)
        outputs = model(seqs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        test_preds.extend(predicted.squeeze().cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds)
print(f"Test F1-score: {test_f1:.4f}")