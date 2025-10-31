"""
LLM Enhancement Layer (Optional)
Converts model output → Actionable solution
"""

import anthropic
import json

class LLMEnhancer:
    """Use LLM to generate deployment guides, Terraform, etc."""
    
    def _init_(self, api_key):
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.enabled = True
        else:
            self.enabled = False
            print("⚠  LLM enhancer disabled (no API key)")
    
    def enhance(self, pipeline_result):
        """
        Enhance pipeline output with LLM-generated content
        
        Args:
            pipeline_result: Output from CompletePipeline
        
        Returns:
            Enhanced result with deployment guide, Terraform, etc.
        """
        if not self.enabled:
            return self._generate_template_output(pipeline_result)
        
        try:
            return self._llm_enhance(pipeline_result)
        except Exception as e:
            print(f"⚠  LLM enhancement failed: {e}")
            return self._generate_template_output(pipeline_result)
    
    def _llm_enhance(self, result):
        """Use Claude to generate actionable output"""
        
        services = [s['service'] for s in result['final_architecture']['predicted_services']]
        requirements = result['document_analysis']['simple_description']
        
        prompt = f"""You are an AWS solutions architect. Generate a complete deployment solution.

REQUIREMENTS:
{requirements}

PREDICTED SERVICES (by ML model with 90% accuracy):
{', '.join(services)}

BUDGET: ${result['metadata']['estimated_monthly_cost']}/month

Generate:

1. *EXECUTIVE SUMMARY* (2-3 sentences)

2. *ARCHITECTURE OVERVIEW* (explain how services work together)

3. *DEPLOYMENT STEPS* (step-by-step CLI commands)

4. *TERRAFORM CODE* (production-ready, with variables)

5. *SECURITY CHECKLIST* (specific to these services)

6. *COST BREAKDOWN* (realistic monthly estimates per service)

7. *MONITORING SETUP* (CloudWatch alarms and metrics)

Be specific and actionable. Include actual commands and code."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        return {
            **result,
            "llm_enhanced_output": {
                "full_text": content,
                "sections": self._parse_sections(content)
            }
        }
    
    def _parse_sections(self, text):
        """Parse LLM output into sections"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            if line.startswith('') and line.endswith(''):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip('*').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _generate_template_output(self, result):
        """Fallback: Generate template-based output"""
        
        services = [s['service'] for s in result['final_architecture']['predicted_services']]
        
        template = f"""
# AWS Architecture Deployment Guide

## Executive Summary
This architecture uses {len(services)} AWS services optimized for your requirements.
Estimated cost: ${result['metadata']['estimated_monthly_cost']}/month

## Services
{', '.join(services)}

## Quick Start

### 1. Set up AWS CLI
bash
aws configure


### 2. Deploy core infrastructure
bash
# Create VPC (if needed)
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Deploy services (use Terraform for production)
terraform init
terraform plan
terraform apply


### 3. Configure services
- Set up IAM roles and policies
- Configure security groups
- Enable CloudWatch monitoring

## Terraform Template
hcl
terraform {{
  required_version = ">= 1.0"
}}

provider "aws" {{
  region = "us-east-1"
}}

# Add resources for: {', '.join(services)}


## Cost Breakdown
Total estimated: ${result['metadata']['estimated_monthly_cost']}/month

## Next Steps
1. Review security settings
2. Set up monitoring and alerts
3. Configure backup and disaster recovery
4. Run load testing
"""
        
        return {
            **result,
            "deployment_guide": template
        }


# Test
if _name_ == "_main_":
    # Mock result for testing
    mock_result = {
        "final_architecture": {
            "predicted_services": [
                {"service": "Lambda", "confidence": 0.95},
                {"service": "DynamoDB", "confidence": 0.92},
                {"service": "S3", "confidence": 0.90}
            ]
        },
        "document_analysis": {
            "simple_description": "Build serverless API for mobile app"
        },
        "metadata": {
            "estimated_monthly_cost": 150
        }
    }
    
    enhancer = LLMEnhancer(api_key=None)  # No API key = template mode
    enhanced = enhancer.enhance(mock_result)
    
    print("✅ LLM Enhancer working!")
    if "deployment_guide" in enhanced:
        print("\n" + enhanced["deployment_guide"][:500] + "...")