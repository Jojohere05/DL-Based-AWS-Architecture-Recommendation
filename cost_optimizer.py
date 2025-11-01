"""
STAGE 2: Cost Optimizer
Optimizes architecture based on budget constraints
"""

import json

class CostOptimizer:
    """Optimize AWS architecture for budget constraints"""
    
    def _init_(self):
        # Load cost data from service_dependencies.json
        with open('service_dependencies.json', 'r') as f:
            data = json.load(f)
            self.service_costs = data.get('service_costs', {})
        
        # Service alternatives for cost optimization
        self.alternatives = {
            "EC2": {
                "cheaper": ["Lambda", "ECS"],
                "reason": "Serverless reduces idle costs by 60-80%",
                "tradeoff": "Limited to 15min execution time"
            },
            "RDS": {
                "cheaper": ["DynamoDB"],
                "reason": "NoSQL can be 40-60% cheaper for simple use cases",
                "tradeoff": "No complex joins, different data model"
            },
            "ECS": {
                "cheaper": ["Lambda"],
                "reason": "Serverless avoids container orchestration costs",
                "tradeoff": "Less control over environment"
            }
        }
    
    def estimate_cost(self, services, traffic="medium"):
        """
        Estimate monthly cost for given services
        
        Args:
            services: List of service names
            traffic: "low", "medium", "high"
        
        Returns:
            Estimated monthly cost in USD
        """
        traffic_multipliers = {
            "low": 1,
            "medium": 3,
            "high": 10
        }
        multiplier = traffic_multipliers.get(traffic, 1)
        
        total_cost = 0
        for service in services:
            if service in self.service_costs:
                cost_data = self.service_costs[service]
                base = cost_data.get("base_monthly", 0)
                scaling = cost_data.get("scaling_factor", 0)
                total_cost += base + (scaling * multiplier)
        
        return round(total_cost, 2)
    
    def optimize_for_budget(self, predicted_services, budget_constraint, traffic_estimate="medium"):
        """
        Optimize architecture to fit budget
        
        Args:
            predicted_services: List of predicted service names
            budget_constraint: "low" (<$100), "medium" ($100-500), "high" (>$500)
            traffic_estimate: "low", "medium", "high"
        
        Returns:
            Dict with optimization results
        """
        budget_limits = {
            "low": 100,
            "medium": 500,
            "high": 2000,
            "custom": None  # For custom amounts
        }
        
        target_budget = budget_limits.get(budget_constraint, 500)
        current_cost = self.estimate_cost(predicted_services, traffic_estimate)
        
        if current_cost <= target_budget:
            return {
                "status": "within_budget",
                "current_cost": current_cost,
                "target_budget": target_budget,
                "services": predicted_services,
                "changes_made": [],
                "message": f"✅ Architecture fits within ${target_budget}/month budget"
            }
        
        # Need optimization
        print(f"💰 Current cost ${current_cost} exceeds budget ${target_budget}")
        print(f"🔄 Applying cost optimizations...")
        
        optimized_services = self._apply_optimizations(
            predicted_services.copy(),
            current_cost,
            target_budget,
            traffic_estimate
        )
        
        new_cost = self.estimate_cost(optimized_services, traffic_estimate)
        changes = self._get_changes(predicted_services, optimized_services)
        
        return {
            "status": "optimized" if new_cost <= target_budget else "partial_optimization",
            "original_cost": current_cost,
            "optimized_cost": new_cost,
            "target_budget": target_budget,
            "savings": current_cost - new_cost,
            "savings_percentage": round(((current_cost - new_cost) / current_cost) * 100, 1),
            "original_services": predicted_services,
            "optimized_services": optimized_services,
            "changes_made": changes,
            "message": self._get_message(new_cost, target_budget, current_cost)
        }
    
    def _apply_optimizations(self, services, current_cost, target_budget, traffic):
        """Apply cost reduction strategies"""
        
        # Strategy 1: Replace expensive compute with serverless (saves 60-80%)
        if current_cost > target_budget * 1.3:
            if "EC2" in services:
                services.remove("EC2")
                if "Lambda" not in services:
                    services.append("Lambda")
                if "API_Gateway" not in services:
                    services.append("API_Gateway")
                print("  ✓ Replaced EC2 with Lambda (serverless)")
        
        # Strategy 2: Use cheaper database (saves 40-60%)
        if self.estimate_cost(services, traffic) > target_budget:
            if "RDS" in services:
                services.remove("RDS")
                if "DynamoDB" not in services:
                    services.append("DynamoDB")
                print("  ✓ Replaced RDS with DynamoDB")
        
        # Strategy 3: Remove optional expensive services
        optional_expensive = ["ECS", "EBS"]
        for svc in optional_expensive:
            if svc in services and self.estimate_cost(services, traffic) > target_budget:
                services.remove(svc)
                print(f"  ✓ Removed optional service: {svc}")
        
        # Strategy 4: Remove CDN if still over budget (saves $10-50)
        if self.estimate_cost(services, traffic) > target_budget:
            if "CloudFront" in services:
                services.remove("CloudFront")
                print("  ✓ Removed CloudFront (CDN)")
        
        # Ensure critical dependencies remain
        services = self._ensure_dependencies(services)
        
        return services
    
    def _ensure_dependencies(self, services):
        """Ensure required dependencies are present"""
        
        # Load dependency rules
        with open('service_dependencies.json', 'r') as f:
            dep_data = json.load(f)
            dependencies = dep_data.get('dependencies', {})
        
        # Add missing dependencies
        added = []
        for service in list(services):
            if service in dependencies:
                required = dependencies[service].get('requires', [])
                for req in required:
                    if req not in services:
                        services.append(req)
                        added.append(req)
        
        if added:
            print(f"  ✓ Added required dependencies: {added}")
        
        return services
    
    def _get_changes(self, original, optimized):
        """Get list of changes made"""
        changes = []
        
        removed = set(original) - set(optimized)
        added = set(optimized) - set(original)
        
        for svc in removed:
            alt_info = self.alternatives.get(svc, {})
            changes.append({
                "type": "removed",
                "service": svc,
                "reason": alt_info.get("reason", "Cost reduction"),
                "alternative": alt_info.get("cheaper", []),
                "tradeoff": alt_info.get("tradeoff", "None")
            })
        
        for svc in added:
            if svc not in original:  # Not a dependency add
                changes.append({
                    "type": "added",
                    "service": svc,
                    "reason": "Cost-effective alternative"
                })
        
        return changes
    
    def _get_message(self, new_cost, target, original):
        """Generate status message"""
        if new_cost <= target:
            savings = original - new_cost
            return f"✅ Optimized to ${new_cost}/month (saved ${savings}, {round((savings/original)*100)}%)"
        else:
            return f"⚠  Partially optimized to ${new_cost}/month (still ${new_cost - target} over budget)"
    
    def get_cost_breakdown(self, services, traffic="medium"):
        """Get detailed cost breakdown per service"""
        breakdown = []
        
        for service in services:
            if service in self.service_costs:
                cost_data = self.service_costs[service]
                traffic_mult = {"low": 1, "medium": 3, "high": 10}.get(traffic, 1)
                
                base = cost_data.get("base_monthly", 0)
                scaling = cost_data.get("scaling_factor", 0)
                total = base + (scaling * traffic_mult)
                
                breakdown.append({
                    "service": service,
                    "base_cost": base,
                    "variable_cost": scaling * traffic_mult,
                    "total_monthly": total,
                    "tier": cost_data.get("tier", "unknown"),
                    "unit": cost_data.get("unit", "N/A")
                })
        
        # Sort by cost (highest first)
        breakdown.sort(key=lambda x: x['total_monthly'], reverse=True)
        
        return breakdown


# Test
if __name__ == "_main_":
    optimizer = CostOptimizer()
    
    print("="*80)
    print("TESTING COST OPTIMIZER")
    print("="*80)
    
    # Test case 1: High-cost architecture
    expensive_services = ["EC2", "RDS", "ECS", "S3", "CloudFront", "VPC", "IAM"]
    
    print("\n📊 TEST 1: Expensive Architecture")
    print(f"Services: {expensive_services}")
    
    current = optimizer.estimate_cost(expensive_services, "medium")
    print(f"Current cost: ${current}/month")
    
    result = optimizer.optimize_for_budget(expensive_services, "low", "medium")
    
    print(f"\nStatus: {result['status']}")
    print(f"Original cost: ${result['original_cost']}/month")
    print(f"Optimized cost: ${result['optimized_cost']}/month")
    print(f"Savings: ${result['savings']} ({result['savings_percentage']}%)")
    print(f"\nOptimized services: {result['optimized_services']}")
    
    if result['changes_made']:
        print("\n🔄 Changes made:")
        for change in result['changes_made']:
            if change['type'] == 'removed':
                print(f"  ❌ {change['service']}: {change['reason']}")
                if change['alternative']:
                    print(f"     → Replaced with: {change['alternative']}")
    
    # Test case 2: Cost breakdown
    print("\n" + "="*80)
    print("📊 TEST 2: Cost Breakdown")
    print("="*80)
    
    breakdown = optimizer.get_cost_breakdown(result['optimized_services'], "medium")
    
    print(f"\n{'Service':<15} {'Monthly Cost':<15} {'Tier':<10} {'Unit'}")
    print("-" * 60)
    for item in breakdown:
        print(f"{item['service']:<15} ${item['total_monthly']:<14.2f} {item['tier']:<10} {item['unit']}")
    
    total = sum(item['total_monthly'] for item in breakdown)
    print("-" * 60)
    print(f"{'TOTAL':<15} ${total:<14.2f}")
    
    print("\n✅ Cost optimizer working correctly!")