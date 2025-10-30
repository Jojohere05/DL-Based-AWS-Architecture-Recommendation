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

# Load services
with open('services.json', 'r') as f:
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

print(f"Label shape: {y.shape}")  # Should be (250, 15)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    df['text'].values, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Save label encoder
import pickle
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("✅ Data loaded and split")