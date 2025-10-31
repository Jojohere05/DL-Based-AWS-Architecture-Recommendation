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
from model_transformer import TransformerServiceClassifier

# Load services
with open('service.json', 'r') as f:
    services_data = json.load(f)

all_services = []
for category, svc_dict in services_data.items():
    all_services.extend(svc_dict.keys())

print(f"Total services: {len(all_services)}")

# Load data (use balanced dataset from Person A)
df = pd.read_csv('data/dataset_balanced.csv')
print(f"Loaded {len(df)} examples")

# Convert labels
df['labels_list'] = df['labels'].apply(lambda x: x.split(','))

mlb = MultiLabelBinarizer(classes=all_services)
y = mlb.fit_transform(df['labels_list'])

print(f"Label shape: {y.shape}")

# Train/val split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    df['text'].values, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# Generate embeddings
print("Loading Sentence-BERT...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings for training set...")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True, batch_size=32)

print("Generating embeddings for validation set...")
X_val_emb = embedder.encode(X_val.tolist(), show_progress_bar=True, batch_size=32)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_emb)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_emb)
y_val_tensor = torch.FloatTensor(y_val)

print("✅ Data prepared")

# Dataset class
class ServiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ServiceDataset(X_train_tensor, y_train_tensor)
val_dataset = ServiceDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TransformerServiceClassifier(
    input_dim=384,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    output_dim=len(all_services),
    dropout=0.3
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in tqdm(loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Validating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Threshold predictions
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='samples', zero_division=0)
    precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='samples', zero_division=0)
    recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='samples', zero_division=0)
    
    return total_loss / len(loader), f1, precision, recall

# Training loop
num_epochs = 40
best_f1 = 0
patience_counter = 0
max_patience = 10

print("\n🚀 Starting training...\n")

history = {
    'train_loss': [],
    'val_loss': [],
    'val_f1': [],
    'val_precision': [],
    'val_recall': []
}

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1, val_precision, val_recall = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_f1)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
        }, 'models/best_transformer_day2.pth')
        print(f"✅ Saved best model (F1: {best_f1:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    print()
    
    # Early stopping
    if patience_counter >= max_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print(f"\n🎉 Training complete!")
print(f"Best F1 Score: {best_f1:.4f}")

# Save training history
import pickle
with open('models/training_history_day2.pkl', 'wb') as f:
    pickle.dump(history, f)