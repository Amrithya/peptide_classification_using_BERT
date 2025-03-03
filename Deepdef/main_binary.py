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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import random
import csv

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
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate, kernel_sizes):
        super(seqClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.kernel_sizes = kernel_sizes  # Store kernel sizes
        
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=self.kernel_sizes[0])
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=self.kernel_sizes[1])
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=self.kernel_sizes[2])
        self.bn3 = nn.BatchNorm1d(256)
        
        self.gru = nn.GRU(256, hidden_dim, batch_first=True, num_layers=num_layers, 
                          bidirectional=True, dropout=dropout_rate)
        
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional GRU
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)))

        if x.shape[-1] < self.kernel_sizes[2]:  # Ensure kernel size is valid
            x = torch.nn.functional.adaptive_avg_pool1d(x, self.kernel_sizes[2])

        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)))

        x, _ = self.gru(x.permute(0, 2, 1))
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

def objective(trial):
    embed_dim = trial.suggest_int("embed_dim", 128, 512, step=64)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    kernel_sizes = [trial.suggest_int(f"kernel_size_{i}", 3, 7) for i in range(3)]

    model = seqClassifier(len(char_to_index) + 1, embed_dim, hidden_dim, num_layers, dropout_rate, kernel_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    for epoch in range(50):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                all_preds.extend(predicted.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.squeeze().cpu().numpy())

        f1 = f1_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn)

        trial.report(f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return f1

"""
Include the following code to run the tuning process:

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
print("Best hyperparameters:", study.best_params)
best_params = study.best_params
kernel_sizes = [
   best_params[f"kernel_size_{i}"] for i in range(3)
]

best_params_without_lr = {
    key: value for key, value in best_params.items() 
    if not key.startswith("kernel_size_") and key != 'learning_rate'
}

model = seqClassifier(
    len(char_to_index) + 1, 
    **best_params_without_lr, 
    kernel_sizes=kernel_sizes
).to(device)

"""

best_params = {
    'embed_dim': 384,
    'hidden_dim': 448,
    'num_layers': 3,
    'dropout_rate': 0.468,
    'kernel_sizes': [6, 6, 6]
}
learning_rate = 1.29e-06

model = seqClassifier(len(char_to_index) + 1, **best_params).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Create a CSV file to log metrics for each epoch
with open('epoch_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "F1-score", "Accuracy", "ROC-AUC", "Sensitivity"])

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
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            all_preds.extend(predicted.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)

    f1_scores.append(f1)
    print(f"Epoch {epoch+1}: F1-score = {f1:.4f}, Accuracy = {accuracy:.4f}, ROC-AUC = {roc_auc:.4f}, Sensitivity = {sensitivity:.4f}")

    # Log metrics for each epoch in the CSV file
    with open('epoch_metrics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, f1, accuracy, roc_auc, sensitivity])

plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), f1_scores, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("F1-score vs Epoch")
plt.grid()
plt.show()

best_epoch = np.argmax(f1_scores) + 1
best_f1 = max(f1_scores)
print(f"Best Epoch: {best_epoch}, Best F1-score: {best_f1:.4f}")

model.eval()
test_preds, test_labels, test_probs = [], [], []
with torch.no_grad():
    for seqs, labels in DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn):
        seqs, labels = seqs.to(device), labels.to(device)
        outputs = model(seqs)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        test_preds.extend(predicted.squeeze().cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probs.squeeze().cpu().numpy())

test_f1 = f1_score(test_labels, test_preds)
test_accuracy = accuracy_score(test_labels, test_preds)
test_roc_auc = roc_auc_score(test_labels, test_probs)
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
test_sensitivity = tp / (tp + fn)

print(f"Test F1-score: {test_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test Sensitivity: {test_sensitivity:.4f}")

# Save test results to CSV
with open('test_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["F1-score", test_f1])
    writer.writerow(["Accuracy", test_accuracy])
    writer.writerow(["ROC-AUC", test_roc_auc])
    writer.writerow(["Sensitivity", test_sensitivity])

# Plot and save ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

def predict(model, input_file, output_file):
    df_test = pd.read_csv(input_file)
    seqs = df_test["sequence"].astype(str).tolist()
    encoded_seqs = [encode_seq(seq) for seq in seqs]
    
    test_loader = DataLoader(encoded_seqs, batch_size=32, shuffle=False, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=0))
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for seqs in test_loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float().cpu().numpy().squeeze()
            predictions.extend(predicted)
    
    df_test["prediction"] = predictions
    df_test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

predict(model, "neoag_test1.csv", "predictions.csv")