"""
Inference Engine for 90% F1 Basic Transformer Model
"""

import torch
import pickle
import json
from sentence_transformers import SentenceTransformer
from model_transformer import TransformerServiceClassifier  # Your original model
import networkx as nx

class BasicArchitectureGenerator:
    """Use your 90% F1 model for predictions"""
    
    def _init_(self, model_path='models/best_transformer_day2.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load checkpoint
        print("📦 Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get service names from label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
        self.service_names = self.mlb.classes_.tolist()
        
        # Initialize model (your basic transformer)
        print("🏗  Initializing model...")
        self.model = TransformerServiceClassifier(
            input_dim=384,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            output_dim=len(self.service_names),
            dropout=0.3
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded (F1: {checkpoint.get('best_f1', 0.90):.4f})")
        
        # Load embedder
        print("📚 Loading Sentence-BERT...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load metadata
        with open('services.json', 'r') as f:
            self.services_data = json.load(f)
        
        with open('service_dependencies.json', 'r') as f:
            self.dep_graph = json.load(f)
        
        print("✅ Inference engine ready!\n")
    
    def predict(self, text, threshold=0.5):
        """
        Predict services from text description
        
        Args:
            text: Requirements description
            threshold: Confidence threshold (0-1)
        
        Returns:
            Dict with predictions
        """
        # Generate embedding
        embedding = self.embedder.encode([text], show_progress_bar=False)
        embedding_tensor = torch.FloatTensor(embedding).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(embedding_tensor)
            outputs = outputs.cpu().numpy()[0]
        
        # Get predictions above threshold
        predicted_indices = outputs > threshold
        predicted_services = [
            self.service_names[i] 
            for i in range(len(self.service_names)) 
            if predicted_indices[i]
        ]
        
        # Add dependencies
        predicted_services = self._add_dependencies(predicted_services)
        
        # Build output
        result = self._build_output(text, predicted_services, outputs)
        
        return result
    
    def _add_dependencies(self, services):
        """Add required dependencies"""
        all_services = set(services)
        dependencies = self.dep_graph.get('dependencies', {})
        
        for service in list(services):
            if service in dependencies:
                required = dependencies[service].get('requires', [])
                all_services.update(required)
        
        return list(all_services)
    
    def _build_output(self, input_text, services, raw_outputs):
        """Build structured output"""
        
        # Get service details
        service_details = []
        for service in services:
            if service not in self.service_names:
                continue
                
            idx = self.service_names.index(service)
            
            # Find category
            category = None
            description = None
            for cat, svc_dict in self.services_data.items():
                if service in svc_dict:
                    category = cat
                    description = svc_dict[service].get('description', 'N/A')
                    break
            
            service_details.append({
                "service": service,
                "confidence": float(raw_outputs[idx]),
                "category": category,
                "description": description
            })
        
        # Sort by confidence
        service_details.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Build graph
        graph = self._build_graph(services)
        
        # Group by category
        categories = {}
        for svc in service_details:
            cat = svc['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(svc['service'])
        
        return {
            "input_text": input_text,
            "predicted_services": service_details,
            "architecture_graph": graph,
            "total_services": len(services),
            "service_categories": categories
        }
    
    def _build_graph(self, services):
        """Build dependency graph"""
        G = nx.DiGraph()
        
        for service in services:
            G.add_node(service)
        
        dependencies = self.dep_graph.get('dependencies', {})
        
        for service in services:
            if service in dependencies:
                deps = dependencies[service]
                
                for dep in deps.get('requires', []):
                    if dep in services:
                        G.add_edge(dep, service, type='requires')
                
                for conn in deps.get('connects_to', []):
                    if conn in services:
                        G.add_edge(service, conn, type='connects_to')
        
        return {
            "nodes": list(G.nodes()),
            "edges": [
                {"from": e[0], "to": e[1], "type": G.edges[e].get('type', 'related')}
                for e in G.edges()
            ]
        }


# Quick test
if __name__ == "_main_":
    generator = BasicArchitectureGenerator()
    
    test_text = "Build a serverless API for mobile app with user authentication"
    
    print("="*80)
    print("TESTING BASIC INFERENCE ENGINE")
    print("="*80)
    print(f"\nInput: {test_text}")
    
    result = generator.predict(test_text)
    
    print(f"\n✅ Predicted {result['total_services']} services:")
    for svc in result['predicted_services'][:5]:
        print(f"  • {svc['service']:15s} ({svc['confidence']:.2%})")
    
    print("\n✅ Inference working!")