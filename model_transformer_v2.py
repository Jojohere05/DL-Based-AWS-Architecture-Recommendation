import torch
import torch.nn as nn
import math
import json

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DependencyAwareTransformer(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_heads=4, num_layers=2, 
                 output_dim=15, dropout=0.3, service_names=None):
        super().__init__()
        
        self.service_names = service_names or []
        
        # Load dependency graph
        with open('service_dependencies.json', 'r') as f:
            self.dep_graph = json.load(f)
        
        # Create dependency matrix
        self.dependency_matrix = self._create_dependency_matrix()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Service embeddings (learn relationships between services)
        self.service_embeddings = nn.Embedding(output_dim, hidden_dim // 2)
        
        # Attention mechanism for service interactions
        self.service_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Confidence head (for output probabilities)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def _create_dependency_matrix(self):
        """Create dependency matrix from JSON"""
        n_services = len(self.service_names)
        dep_matrix = torch.zeros(n_services, n_services)
        
        for i, service in enumerate(self.service_names):
            if service in self.dep_graph['dependencies']:
                deps = self.dep_graph['dependencies'][service].get('requires', [])
                for dep in deps:
                    if dep in self.service_names:
                        j = self.service_names.index(dep)
                        dep_matrix[i, j] = 1.0
        
        return dep_matrix
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # x shape: (batch_size, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Project and add positional encoding
        x = self.input_projection(x)  # (batch_size, 1, hidden_dim)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_dim)
        
        # Get service embeddings
        service_ids = torch.arange(len(self.service_names), device=x.device)
        service_emb = self.service_embeddings(service_ids)  # (num_services, hidden_dim//2)
        service_emb = service_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_services, hidden_dim//2)
        
        # Expand x to match service dimensions
        x_expanded = x.expand(-1, len(self.service_names), -1)  # (batch_size, num_services, hidden_dim)
        
        # Concatenate text features with service embeddings
        combined = torch.cat([x_expanded, service_emb], dim=-1)  # (batch_size, num_services, hidden_dim + hidden_dim//2)
        
        # Predict for each service
        logits = self.classifier(combined)  # (batch_size, num_services, num_services)
        
        # Take diagonal (each service's own prediction)
        output = torch.diagonal(logits, dim1=1, dim2=2)  # (batch_size, num_services)
        
        # Confidence scores
        confidence = self.confidence_head(x.squeeze(1))  # (batch_size, num_services)
        
        return output, confidence
    
    def compute_dependency_loss(self, predictions, targets):
        """
        Penalize predictions that violate dependencies
        If service A is predicted but required service B is not, add penalty
        """
        device = predictions.device
        dep_matrix = self.dependency_matrix.to(device)
        
        batch_size = predictions.size(0)
        
        # For each predicted service, check if dependencies are also predicted
        # predictions shape: (batch_size, num_services)
        # dep_matrix shape: (num_services, num_services)
        
        # dep_matrix[i, j] = 1 means service i requires service j
        
        violation_penalty = 0.0
        
        for i in range(len(self.service_names)):
            # Get services that service i depends on
            dependencies = dep_matrix[i]  # (num_services,)
            
            # If service i is predicted
            service_i_predicted = predictions[:, i]  # (batch_size,)
            
            # Check if dependencies are predicted
            for j in range(len(self.service_names)):
                if dependencies[j] > 0:  # j is a dependency of i
                    service_j_predicted = predictions[:, j]
                    
                    # Penalty: service i predicted but dependency j not predicted
                    violation = service_i_predicted * (1 - service_j_predicted)
                    violation_penalty += violation.mean()
        
        return violation_penalty

# Test
if __name__ == "__main__":
    service_names = ["EC2", "Lambda", "S3", "RDS", "DynamoDB", "VPC", "IAM", 
                     "API_Gateway", "CloudFront", "Cognito", "SNS", "SQS", "ECS", "EBS"]
    
    model = DependencyAwareTransformer(
        input_dim=384,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        output_dim=len(service_names),
        dropout=0.3,
        service_names=service_names
    )
    
    batch = torch.randn(4, 384)
    output, confidence = model(batch)
    print(f"Output shape: {output.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"✅ Model created successfully")