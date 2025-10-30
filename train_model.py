import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import pickle
from sklearn.metrics import f1_score, classification_report

# Load services
with open('service.json', 'r') as f:
    services_data = json.load(f)

all_services = []
for category, svc_dict in services_data.items():
    all_services.extend(svc_dict.keys())

print(f"Total services: {len(all_services)}")

# Load full dataset
df = pd.read_csv('data/dataset_day1.csv')
print(f"Loaded {len(df)} total examples")

# Create labels_list column on full df before fitting label binarizer
df['labels_list'] = df['labels'].apply(lambda x: x.split(','))

# Create test set (10%) - holds out separate test data from full dataset
df_train_val, df_test = train_test_split(df, test_size=0.10, random_state=42)

print(f"Train+Val examples: {len(df_train_val)}, Test examples: {len(df_test)}")

# Prepare labels for train_val and test (labels_list already exists)
y_train_val = None
y_test = None

# Initialize MultiLabelBinarizer, fit on full dataset labels for consistent encoding
mlb = MultiLabelBinarizer(classes=all_services)
mlb.fit(df['labels_list'])

# Encode labels for train_val and test sets
y_train_val = mlb.transform(df_train_val['labels_list'])
y_test = mlb.transform(df_test['labels_list'])

print(f"Train+Val label shape: {y_train_val.shape}, Test label shape: {y_test.shape}")

# Split train_val into train and val sets (80/20)
X_train_val = df_train_val['text'].values
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Save label encoder for inference
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("✅ Label encoder saved")

# Load sentence transformer for embeddings generation
print("Loading Sentence-BERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Generate embeddings for train, val, and test
print("Generating embeddings for training set...")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True, batch_size=32)

print("Generating embeddings for validation set...")
X_val_emb = embedder.encode(X_val.tolist(), show_progress_bar=True, batch_size=32)

print("Generating embeddings for test set...")
X_test_emb = embedder.encode(df_test['text'].tolist(), show_progress_bar=True, batch_size=32)

print(f"Embedding shapes - Train: {X_train_emb.shape}, Val: {X_val_emb.shape}, Test: {X_test_emb.shape}")

# Convert embeddings and labels to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_emb)
y_train_tensor = torch.FloatTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val_emb)
y_val_tensor = torch.FloatTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test_emb)
y_test_tensor = torch.FloatTensor(y_test)

print("✅ Embeddings generated and tensors created")

# Define the MLP model
class ServiceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

input_dim = 384
hidden_dim = 256
output_dim = len(all_services)

model = ServiceClassifier(input_dim, hidden_dim, output_dim)
print(f"✅ Model created with parameters: {sum(p.numel() for p in model.parameters())}")

# Setup training
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dataset class
class ServiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Data loaders
train_loader = DataLoader(ServiceDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
val_loader = DataLoader(ServiceDataset(X_val_tensor, y_val_tensor), batch_size=16, shuffle=False)
test_loader = DataLoader(ServiceDataset(X_test_tensor, y_test_tensor), batch_size=16, shuffle=False)

# Training and validation functions
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.append(preds)
            all_labels.append(y_batch)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='samples', zero_division=0)
    return total_loss / len(loader), f1

# Train loop
num_epochs = 30
best_f1 = 0

print("\n🚀 Starting training...\n")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_f1 = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'models/best_model_day1.pth')
        print(f"  ✅ Saved best model (F1: {best_f1:.4f})")
    print()

print(f"🎉 Training complete! Best Val F1: {best_f1:.4f}")

# Load best model for testing
model.load_state_dict(torch.load('models/best_model_day1.pth'))
model.eval()

# Test evaluation on independent test set
print("\n🧪 Evaluating generalization on unseen test set...\n")

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = (outputs > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("Overall test F1 score (samples average):", f1_score(all_labels, all_preds, average='samples'))
print("\nDetailed classification report:")
print(classification_report(all_labels, all_preds, target_names=all_services, zero_division=0))

# Example test predictions
test_texts = [
    "Build a web application with user authentication and file storage",
    "Create a serverless API for mobile app",
    "Develop a data processing pipeline for logs"
]

print("\n🧪 Example test predictions:\n")

test_emb = embedder.encode(test_texts)
test_tensor = torch.FloatTensor(test_emb)

with torch.no_grad():
    predictions = model(test_tensor)
    pred_labels = (predictions > 0.5).numpy()

for i, text in enumerate(test_texts):
    print(f"Input: {text}")
    predicted_services = mlb.inverse_transform(pred_labels[i:i+1])[0]
    print(f"Predicted services: {list(predicted_services)}\n")
