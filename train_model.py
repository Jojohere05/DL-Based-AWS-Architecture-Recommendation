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
from sklearn.metrics import f1_score

# Load services
with open('service.json', 'r') as f:
    services_data = json.load(f)

all_services = []
for category, svc_dict in services_data.items():
    all_services.extend(svc_dict.keys())

print(f"Total services: {len(all_services)}")

# Load data
df = pd.read_csv('data/dataset_day1.csv')
print(f"Loaded {len(df)} examples")

# Convert labels to multi-hot encoding
df['labels_list'] = df['labels'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer(classes=all_services)
y = mlb.fit_transform(df['labels_list'])

print(f"Label shape: {y.shape}")  # Should be (number_of_samples, number_of_services)

# Show example of label binarization output
print("\nExample label binarization:")
for i in range(3):
    print(f"Labels: {df['labels_list'].iloc[i]}")
    print(f"Encoded: {y[i]}")

# Train/val split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    df['text'].values, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Save label encoder for inference later
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("✅ Label encoder saved")

# Load sentence transformer
print("Loading Sentence-BERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# Generate embeddings
print("Generating embeddings for training set...")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True, batch_size=32)

print("Generating embeddings for validation set...")
X_val_emb = embedder.encode(X_val.tolist(), show_progress_bar=True, batch_size=32)

print(f"Embedding shape (train): {X_train_emb.shape}")  # Should be (train_samples, 384)
print(f"Embedding shape (val): {X_val_emb.shape}")      # Should be (val_samples, 384)

# Show example of embedding output
print("\nExample training embedding vector (first training sample):")
print(X_train_emb[0])

# Convert embeddings and labels to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_emb)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_emb)
y_val_tensor = torch.FloatTensor(y_val)

print("✅ Embeddings generated and converted to tensors")

# Simple MLP model definition
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

# Initialize model
input_dim = 384  # Sentence-BERT embedding size
hidden_dim = 256
output_dim = len(all_services)

model = ServiceClassifier(input_dim, hidden_dim, output_dim)
print(f"✅ Model created: {input_dim} -> {hidden_dim} -> {output_dim}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dataset class
class ServiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loaders
train_dataset = ServiceDataset(X_train_tensor, y_train_tensor)
val_dataset = ServiceDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training function
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

# Validation function
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

print(f"🎉 Training complete! Best F1: {best_f1:.4f}")

# Load best model for testing
model.load_state_dict(torch.load('models/best_model_day1.pth'))
model.eval()

# Test on a few examples
test_texts = [
    "Build a web application with user authentication and file storage",
    "Create a serverless API for mobile app",
    "Develop a data processing pipeline for logs"
]

print("\n🧪 Testing predictions:\n")

test_emb = embedder.encode(test_texts)
test_tensor = torch.FloatTensor(test_emb)

with torch.no_grad():
    predictions = model(test_tensor)
    pred_labels = (predictions > 0.5).numpy()

for i, text in enumerate(test_texts):
    print(f"Input: {text}")
    predicted_services = mlb.inverse_transform(pred_labels[i:i+1])[0]
    print(f"Predicted services: {list(predicted_services)}")
    print()
