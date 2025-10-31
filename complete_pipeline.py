"""
COMPLETE PIPELINE: Document → Your 90% Model → Enhanced Output
"""

import json
import os
from document_parser import RequirementsDocumentParser
from inference_basic import BasicArchitectureGenerator
from cost_optimizer import CostOptimizer

class CompletePipeline:
    """
    Full pipeline integrating:
    1. Document parsing (LLM pre-processor)
    2. Your 90% transformer model
    3. Cost optimization (rule-based post-processor)
    4. LLM enhancement (optional)
    """
    
    def _init_(self, anthropic_api_key=None):
        print("🚀 Initializing Complete Pipeline...")
        print("="*80)
        
        # Stage 1: Document Parser (with or without LLM)
        self.parser = RequirementsDocumentParser(api_key=anthropic_api_key)
        
        # Stage 2: Your 90% Model
        self.generator = BasicArchitectureGenerator()
        
        # Stage 3: Cost Optimizer
        self.optimizer = CostOptimizer()
        
        print("="*80)
        print("✅ Pipeline ready!\n")
    
    def process(self, document_text, budget_constraint="medium", use_llm_enhancement=False):
        """
        Complete processing pipeline
        
        Args:
            document_text: Raw document (can be complex)
            budget_constraint: "low", "medium", "high" or custom amount
            use_llm_enhancement: Whether to use LLM for final output
        
        Returns:
            Complete architecture solution
        """
        
        print("🔄 STAGE 1: Parsing Document")
        print("-" * 80)
        
        # Parse document
        parsed = self.parser.parse_document(document_text)
        
        print(f"✓ Extracted description: {parsed['simple_description'][:100]}...")
        print(f"✓ Scale: {parsed['scale_indicators']}")
        print(f"✓ Features: {parsed['technical_features'][:5]}")
        
        print("\n🔄 STAGE 2: Predicting Services (Your 90% Model)")
        print("-" * 80)
        
        # Use simple description for your model
        base_prediction = self.generator.predict(
            parsed['simple_description'],
            threshold=0.5
        )
        
        print(f"✓ Predicted {base_prediction['total_services']} services")
        print(f"✓ Top services: {[s['service'] for s in base_prediction['predicted_services'][:5]]}")
        
        print("\n🔄 STAGE 3: Post-Processing")
        print("-" * 80)
        
        # Enhance with document features
        enhanced_services = self._enhance_with_features(
            [s['service'] for s in base_prediction['predicted_services']],
            parsed['technical_features'],
            parsed['compute_preference']
        )
        
        print(f"✓ Enhanced to {len(enhanced_services)} services")
        
        # Determine traffic tier from scale
        scale = parsed['scale_indicators']
        if scale['users']:
            if scale['users'] < 10000:
                traffic = "low"
            elif scale['users'] < 50000:
                traffic = "medium"
            else:
                traffic = "high"
        else:
            traffic = "medium"
        
        # Convert budget constraint
        if isinstance(budget_constraint, int):
            if budget_constraint < 100:
                budget_tier = "low"
            elif budget_constraint < 500:
                budget_tier = "medium"
            else:
                budget_tier = "high"
        else:
            budget_tier = budget_constraint
        
        # Apply cost optimization
        print(f"✓ Optimizing for budget: {budget_tier} ({traffic} traffic)")
        
        cost_result = self.optimizer.optimize_for_budget(
            enhanced_services,
            budget_tier,
            traffic
        )
        
        print(f"✓ {cost_result['status']}")
        if cost_result['status'] == 'optimized':
            print(f"✓ Cost reduced from ${cost_result['original_cost']} to ${cost_result['optimized_cost']}")
        
        print("\n🔄 STAGE 4: Building Final Output")
        print("-" * 80)
        
        # Get final service list
        final_services = cost_result.get('optimized_services', cost_result.get('services', enhanced_services))
        
        # Rebuild architecture with final services
        final_architecture = self.generator.predict(
            parsed['simple_description'],
            threshold=0.3  # Lower threshold to ensure we get all needed services
        )
        
        # Filter to only final services
        final_architecture['predicted_services'] = [
            s for s in final_architecture['predicted_services']
            if s['service'] in final_services
        ]
        final_architecture['total_services'] = len(final_services)
        
        # Build complete result
        result = {
            "document_analysis": parsed,
            "model_prediction": base_prediction,
            "cost_optimization": cost_result,
            "final_architecture": final_architecture,
            "metadata": {
                "model_used": "TransformerServiceClassifier (F1: 90.52%)",
                "budget_tier": budget_tier,
                "traffic_estimate": traffic,
                "estimated_monthly_cost": cost_result.get('optimized_cost', cost_result.get('current_cost', 0))
            }
        }
        
        print("✅ Pipeline complete!")
        
        return result
    
    def _enhance_with_features(self, base_services, features, compute_pref):
        """
        Add services based on document features
        """
        enhanced = set(base_services)
        
        # Video streaming
        if "video_streaming" in features:
            enhanced.update(["S3", "CloudFront"])
        
        # AI/ML
        if "ai_ml" in features:
            if compute_pref == "gpu":
                enhanced.add("EC2")
            elif compute_pref == "serverless":
                enhanced.add("Lambda")
        
        # Real-time
        if "real_time" in features:
            enhanced.add("API_Gateway")
        
        # Authentication
        if "authentication" in features:
            enhanced.add("Cognito")
        
        # File uploads
        if "file_upload" in features:
            enhanced.add("S3")
        
        # Database
        if "database" in features:
            if "DynamoDB" not in enhanced and "RDS" not in enhanced:
                enhanced.add("DynamoDB")
        
        # Messaging
        if "messaging" in features:
            if "SNS" not in enhanced and "SQS" not in enhanced:
                enhanced.add("SNS")
        
        # Event-driven
        if "event_driven" in features:
            enhanced.update(["Lambda", "SQS"])
        
        # Always ensure IAM
        enhanced.add("IAM")
        
        # Add VPC if needed
        if any(svc in enhanced for svc in ["EC2", "RDS", "ECS"]):
            enhanced.add("VPC")
        
        return list(enhanced)
    
    def save_result(self, result, output_file):
        """Save result to JSON"""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved to: {output_file}")


# Test with your EdTech example
if __name__ == "_main_":
    
    # Test document
    edtech_doc = """
    Example document for an EdTech platform:
    
    Project Name: EduVision — Online Coaching for Entrance Exams
    Prepared for: Founders & Tech Team
    Date: September 2025
    
    1. Overview
    EduVision is an EdTech startup aimed at providing live interactive classes and 
    recorded lectures for students preparing for national-level entrance exams. The 
    platform will host thousands of students simultaneously, especially during peak 
    admission months.
    
    We aim to deliver seamless learning experiences through video streaming, AI-
    driven personalized course recommendations, and secure student data management.
    
    Expected Scale:
    • Daily active users: ~50,000
    • Peak concurrency: ~3,000 users at the same time
    • Storage needs: ~20 TB (primarily lecture videos and transcripts)
    • Monthly infra budget: ~$8,000
    
    3. Non-Functional Requirements
    • Performance: API response latency should remain under 150 ms for quizzes and 
      interactions. Video startup latency should be <300 ms globally.
    • Availability: 99.9% uptime, especially during live classes.
    • Integration: Event-driven processing for handling lecture uploads, automatic 
      transcript generation, and indexing.
    • Messaging: Asynchronous communication required for scheduling tasks, 
      notifications, and event pipelines.
    
    4. Data & AI Needs
    • Lecture videos (~20 TB) stored with long-term archival of past years' content.
    • NLP pipeline required to process lecture transcripts and index them for search.
    • Recommender system to suggest personalized content → requires medium-to-large 
      AI model training on student behavior data.
    • Training workloads are compute-intensive and will occasionally require GPU-based 
      infrastructure.
    """
    
    print("="*80)
    print("TESTING COMPLETE PIPELINE WITH YOUR 90% MODEL")
    print("="*80)
    print()
    
    # Initialize pipeline (no API key = uses fallback parser)
    pipeline = CompletePipeline(anthropic_api_key=None)
    
    # Process document
    result = pipeline.process(
        edtech_doc,
        budget_constraint="high",  # $8000 budget
        use_llm_enhancement=False
    )
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print(f"\n📊 Final Architecture:")
    print(f"  Services: {result['final_architecture']['total_services']}")
    print(f"  Estimated Cost: ${result['metadata']['estimated_monthly_cost']}/month")
    
    print(f"\n🔧 Services by Category:")
    for category, services in result['final_architecture']['service_categories'].items():
        print(f"  {category.upper()}: {', '.join(services)}")
    
    print(f"\n💰 Cost Optimization:")
    cost_opt = result['cost_optimization']
    if cost_opt['status'] == 'optimized':
        print(f"  Original: ${cost_opt['original_cost']}/month")
        print(f"  Optimized: ${cost_opt['optimized_cost']}/month")
        print(f"  Savings: ${cost_opt['savings']} ({cost_opt['savings_percentage']}%)")
    else:
        print(f"  Status: {cost_opt['message']}")
    
    # Save result
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    pipeline.save_result(result, f'{output_dir}/edtech_complete_architecture.json')
    
    print("\n✅ Complete pipeline test successful!")