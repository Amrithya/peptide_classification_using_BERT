import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
import numpy as np
import optuna  # For hyperparameter tuning
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Character encoding dictionary
char_to_index = {ch: i+1 for i, ch in enumerate("ARNDBCEQZGHILKMFPSTWYVXOJU")}

def encode_sequence(seq, max_len=150):
    encoded = [char_to_index.get(ch, 0) for ch in seq]
    encoded = encoded[:max_len] + [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)

class PeptideDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data['sequence'].apply(lambda x: encode_sequence(x)).tolist()
        self.labels = self.data['label'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Define the GRU Model with tunable hyperparameters
class GRUModel(nn.Module):
    def __init__(self, num_filters, kernel_size, stride, gru_hidden_size, dropout_rate):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(len(char_to_index) + 1, 28)
        self.conv = nn.Conv1d(28, num_filters, kernel_size=kernel_size, stride=stride)
        conv_output_size = ((150 - kernel_size) // stride) + 1

        self.gru = nn.GRU(num_filters, gru_hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(gru_hidden_size * 2 * conv_output_size, 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.leaky_relu(self.conv(x)).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(1)

# Load dataset
dataset = PeptideDataset("neoag_train.csv")

# Split dataset
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

def objective(trial):
    num_filters = trial.suggest_int("num_filters", 16, 128)
    kernel_size = trial.suggest_int("kernel_size", 2, 9)
    stride = trial.suggest_int("stride", 1, 3)
    gru_hidden_size = trial.suggest_int("gru_hidden_size", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUModel(num_filters, kernel_size, stride, gru_hidden_size, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(5):
        for sequences, labels in train_loader:
            sequences, labels = sequences.long(), labels.float()
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.long(), labels.float()
            outputs = model(sequences)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            y_prob.extend(outputs.tolist())

    f1 = f1_score(y_true, y_pred)
    return f1  # Optimize for F1-score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10) 

best_params = study.best_params
print("Best Hyperparameters:", best_params)

def train_final_model():
    best_model = GRUModel(
        best_params["num_filters"],
        best_params["kernel_size"],
        best_params["stride"],
        best_params["gru_hidden_size"],
        best_params["dropout_rate"]
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    metrics = []

    with open("training_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "F1-score", "Accuracy", "ROC-AUC", "Sensitivity"])

        best_model.train()
        for epoch in range(50):
            total_loss = 0
            for sequences, labels in train_loader:
                sequences, labels = sequences.long(), labels.float()
                optimizer.zero_grad()
                outputs = best_model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluation on validation set
            best_model.eval()
            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.long(), labels.float()
                    outputs = best_model(sequences)
                    predicted = (outputs > 0.5).float()
                    y_true.extend(labels.tolist())
                    y_pred.extend(predicted.tolist())
                    y_prob.extend(outputs.tolist())

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_prob)

            metrics.append([epoch + 1, f1, accuracy, roc_auc, recall])
            writer.writerow([epoch + 1, f1, accuracy, roc_auc, recall])

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, Sensitivity: {recall:.4f}")
    
    return best_model, metrics, y_true, y_prob

def plot_metrics(metrics):
    epochs = [m[0] for m in metrics]
    f1_scores = [m[1] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, f1_scores, label="F1-score", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("F1-score over Epochs")
    plt.savefig("training_metrics_plot.png")
    plt.show()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()

final_model, metrics, y_true, y_prob = train_final_model()
plot_metrics(metrics)
plot_roc_curve(y_true, y_prob)

def evaluate_model(model):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.long()
            outputs = model(sequences)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            y_prob.extend(outputs.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"Test Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {roc_auc:.4f}")

evaluate_model(final_model)