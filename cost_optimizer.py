"""
Cost Optimizer - Apply budget constraints and optimization rules
"""

import json

class CostOptimizer:
    """Optimize service recommendations based on budget and scale"""
    
    def __init__(self):
        # Load cost estimates (approximate monthly costs)
        self.cost_estimates = {
            "Lambda": 10,
            "DynamoDB": 25,
            "S3": 15,
            "API_Gateway": 20,
            "CloudFront": 50,
            "Cognito": 10,
            "SNS": 5,
            "SQS": 5,
            "IAM": 0,
            "VPC": 0,
            "EC2": 150,
            "RDS": 200,
            "ECS": 100,
            "EBS": 30
        }
        
        # Service alternatives (cheaper → more expensive)
        self.alternatives = {
            "RDS": "DynamoDB",  # NoSQL is cheaper
            "EC2": "Lambda",    # Serverless is cheaper
            "ECS": "Lambda"     # Serverless containers
        }
    
    def optimize_for_budget(self, services, budget_tier, traffic_level="medium"):
        """
        Optimize service list for budget constraints
        
        Args:
            services: List of service names
            budget_tier: "low" (<$100), "medium" ($100-500), "high" (>$500)
            traffic_level: "low", "medium", "high"
        
        Returns:
            Dict with optimization results
        """
        
        # Calculate current cost
        current_cost = sum(self.cost_estimates.get(svc, 0) for svc in services)
        
        # Budget thresholds
        budgets = {
            "low": 100,
            "medium": 500,
            "high": 2000
        }
        
        budget_limit = budgets.get(budget_tier, 500)
        
        # Check if within budget
        if current_cost <= budget_limit:
            return {
                "status": "within_budget",
                "services": services,
                "current_cost": current_cost,
                "budget_limit": budget_limit,
                "message": f"Architecture fits within {budget_tier} budget"
            }
        
        # Need to optimize
        print(f"⚠️  Current cost ${current_cost} exceeds {budget_tier} budget ${budget_limit}")
        print(f"   Applying optimizations...")
        
        optimized_services = list(services)
        optimizations_applied = []
        
        # Apply alternatives
        for expensive, cheaper in self.alternatives.items():
            if expensive in optimized_services and current_cost > budget_limit:
                optimized_services.remove(expensive)
                if cheaper not in optimized_services:
                    optimized_services.append(cheaper)
                
                cost_reduction = self.cost_estimates.get(expensive, 0) - self.cost_estimates.get(cheaper, 0)
                current_cost -= cost_reduction
                
                optimizations_applied.append(f"Replaced {expensive} with {cheaper} (saved ${cost_reduction})")
                print(f"   ✓ Replaced {expensive} → {cheaper}")
        
        # Remove optional services if still over budget
        optional_services = ["CloudFront", "ECS", "EBS"]
        for optional in optional_services:
            if optional in optimized_services and current_cost > budget_limit:
                cost_reduction = self.cost_estimates.get(optional, 0)
                optimized_services.remove(optional)
                current_cost -= cost_reduction
                
                optimizations_applied.append(f"Removed optional {optional} (saved ${cost_reduction})")
                print(f"   ✓ Removed {optional}")
        
        # Calculate savings
        original_cost = sum(self.cost_estimates.get(svc, 0) for svc in services)
        optimized_cost = sum(self.cost_estimates.get(svc, 0) for svc in optimized_services)
        savings = original_cost - optimized_cost
        savings_pct = int((savings / original_cost) * 100) if original_cost > 0 else 0
        
        return {
            "status": "optimized",
            "services": services,
            "optimized_services": optimized_services,
            "original_cost": original_cost,
            "optimized_cost": optimized_cost,
            "budget_limit": budget_limit,
            "savings": savings,
            "savings_percentage": savings_pct,
            "optimizations_applied": optimizations_applied,
            "message": f"Reduced cost from ${original_cost} to ${optimized_cost}"
        }


# Test
if __name__ == "__main__":
    optimizer = CostOptimizer()
    
    services = ["EC2", "RDS", "S3", "CloudFront", "IAM", "VPC"]
    
    result = optimizer.optimize_for_budget(services, "low", "medium")
    
    print(json.dumps(result, indent=2))
    print("\n✅ Optimizer working!")
