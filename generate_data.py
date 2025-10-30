import json
import random
import pandas as pd
from collections import Counter

# Load services
with open('service.json', 'r') as f:
    services = json.load(f)

# Flatten services list
all_services = []
for category, svc_dict in services.items():
    all_services.extend(svc_dict.keys())

print(f"Total services: {len(all_services)}")

# Templates for different use cases
templates = {
    "web_application": [
        "Build a {scale} web application with {auth} and {storage}",
        "Create a {framework} website that needs {database} and {cdn}",
        "Develop a web platform for {purpose} with {features}",
    ],
    "api_service": [
        "Build a {type} API for {purpose} with {database}",
        "Create a RESTful API that handles {feature} and stores data in {database}",
        "Develop a {scale} API gateway for {purpose}",
    ],
    "data_processing": [
        "Build a data pipeline to process {data_type} and store in {storage}",
        "Create a {frequency} batch processing system for {purpose}",
        "Develop a streaming data platform for {use_case}",
    ],
    "mobile_backend": [
        "Build a backend for a mobile app that needs {features}",
        "Create a {scale} mobile API with {auth} and {notifications}",
        "Develop a mobile backend for {app_type} with {database}",
    ],
    "serverless": [
        "Build a serverless application for {purpose}",
        "Create an event-driven system that {action}",
        "Develop a serverless API for {use_case}",
    ],
    "static_website": [
        "Host a simple static website for {purpose}",
        "Deploy a landing page with {features}",
    ],
    "authentication_system": [
        "Build a user authentication system with {auth_method}",
        "Create a login system for {user_type}",
    ],
    "file_processing": [
        "Process {file_type} files uploaded by users",
        "Build a file conversion service for {conversion}",
    ],
    "real_time_app": [
        "Build a real-time {app_type} application",
        "Create a live {feature} system",
    ],
    "analytics_platform": [
        "Build an analytics dashboard for {data_source}",
        "Create a reporting system for {metrics}",
    ]
}

# Substitution options
substitutions = {
    "scale": ["small", "medium", "large-scale", "high-traffic"],
    "auth": ["user authentication", "login system", "OAuth", "social login"],
    "storage": ["file uploads", "media storage", "document storage", "image storage"],
    "framework": ["React", "Vue.js", "Angular", "Next.js"],
    "database": ["user data", "product catalog", "transaction history", "analytics data"],
    "cdn": ["global content delivery", "fast page loads", "edge caching"],
    "purpose": ["e-commerce", "social media", "booking system", "CRM", "inventory management"],
    "features": ["real-time updates", "file sharing", "chat functionality", "payment processing"],
    "type": ["RESTful", "GraphQL", "serverless", "microservices"],
    "data_type": ["logs", "images", "videos", "CSV files", "sensor data"],
    "frequency": ["hourly", "daily", "real-time", "batch"],
    "use_case": ["IoT devices", "analytics", "machine learning", "ETL"],
    "notifications": ["push notifications", "email alerts", "SMS notifications"],
    "app_type": ["social networking", "e-commerce", "fitness tracking", "messaging"],
    "action": ["processes uploads", "sends notifications", "generates reports", "handles webhooks"],
    "auth_method": ["email/password", "social login", "SSO", "multi-factor authentication"],
    "user_type": ["customers", "employees", "partners", "students"],
    "file_type": ["PDF", "image", "video", "CSV", "Excel"],
    "conversion": ["PDF to text", "image resize", "video transcoding"],
    "feature": ["updates", "messaging", "notifications", "tracking"],
    "data_source": ["website traffic", "sales data", "user behavior", "IoT sensors"],
    "metrics": ["revenue", "user engagement", "system performance"],
}

# Ground truth labels for each use case type
use_case_labels = {
    "web_application": ["EC2", "VPC", "IAM", "S3", "RDS", "CloudFront"],
    "api_service": ["API_Gateway", "Lambda", "IAM", "DynamoDB", "CloudFront"],
    "data_processing": ["Lambda", "S3", "IAM", "SQS"],
    "mobile_backend": ["API_Gateway", "Lambda", "DynamoDB", "Cognito", "SNS", "IAM"],
    "serverless": ["Lambda", "API_Gateway", "DynamoDB", "IAM", "S3"],
    "static_website": ["S3", "CloudFront", "IAM"],
    "authentication_system": ["Cognito", "Lambda", "IAM", "DynamoDB"],
    "file_processing": ["S3", "Lambda", "IAM", "SQS"],
    "real_time_app": ["API_Gateway", "Lambda", "DynamoDB", "IAM"],
    "analytics_platform": ["S3", "Lambda", "RDS", "IAM"],
}

def generate_example(use_case_type):
    """Generate one training example"""
    template = random.choice(templates[use_case_type])
    text = template
    for placeholder, options in substitutions.items():
        if f"{{{placeholder}}}" in text:
            text = text.replace(f"{{{placeholder}}}", random.choice(options))
    labels = use_case_labels.get(use_case_type, []).copy()
    if random.random() > 0.7:
        extra_services = list(set(all_services) - set(labels))
        if extra_services:
            labels.append(random.choice(extra_services))
    if random.random() > 0.8 and len(labels) > 3:
        labels.remove(random.choice(labels))
    return {
        "text": text,
        "labels": ",".join(labels),
        "use_case": use_case_type
    }

def generate_dataset(total_samples=450):
    """Generate dataset with balanced use case distribution"""
    data = []
    samples_per_use_case = total_samples // len(templates)
    for use_case_type in templates.keys():
        for _ in range(samples_per_use_case):
            data.append(generate_example(use_case_type))
    remaining = total_samples - len(data)
    for _ in range(remaining):
        use_case_type = random.choice(list(templates.keys()))
        data.append(generate_example(use_case_type))
    return pd.DataFrame(data)

# Generate main dataset
print("Generating 450 examples...")
df = generate_dataset(450)

# ECS and EBS examples
ecs_examples = [
    {"text": "Deploy a containerized microservices application", "labels": "ECS,VPC,IAM", "use_case": "web_application"},
    {"text": "Run Docker containers for my web application", "labels": "ECS,VPC,IAM,RDS", "use_case": "web_application"},
    {"text": "Build a Kubernetes-based application with multiple services", "labels": "ECS,VPC,IAM,RDS,S3", "use_case": "web_application"},
    {"text": "Deploy containerized microservices architecture", "labels": "ECS,VPC,IAM,DynamoDB", "use_case": "web_application"},
    {"text": "Host Docker containers for data processing jobs", "labels": "ECS,VPC,IAM,S3", "use_case": "data_processing"},
    {"text": "Run scalable containerized API services", "labels": "ECS,API_Gateway,VPC,IAM", "use_case": "api_service"},
    {"text": "Deploy multiple Docker images for different services", "labels": "ECS,VPC,IAM,CloudFront", "use_case": "web_application"},
    {"text": "Build container orchestration platform", "labels": "ECS,VPC,IAM,RDS", "use_case": "web_application"},
    {"text": "Deploy microservices using Docker Compose", "labels": "ECS,VPC,IAM,DynamoDB,S3", "use_case": "web_application"},
    {"text": "Run containerized batch processing jobs", "labels": "ECS,S3,VPC,IAM,SQS", "use_case": "data_processing"},
    {"text": "Host multiple containerized applications", "labels": "ECS,VPC,IAM,RDS,CloudFront", "use_case": "web_application"},
    {"text": "Build a container-based analytics platform", "labels": "ECS,VPC,IAM,RDS,S3", "use_case": "web_application"},
]

ebs_examples = [
    {"text": "Attach persistent disk storage to virtual machine", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Need block storage volumes for database server", "labels": "EC2,EBS,VPC,IAM,RDS", "use_case": "database"},
    {"text": "Add persistent disk space to EC2 instance", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Attach SSD storage to application server", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Mount persistent volumes for file system", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Need high-performance disk storage for server", "labels": "EC2,EBS,VPC,IAM", "use_case": "compute"},
    {"text": "Attach storage volumes to compute instances", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Provision block storage for virtual machines", "labels": "EC2,EBS,VPC,IAM", "use_case": "web_application"},
    {"text": "Add disk volumes to running instances", "labels": "EC2,EBS,VPC,IAM", "use_case": "compute"},
    {"text": "Create persistent storage for EC2 workloads", "labels": "EC2,EBS,VPC,IAM,S3", "use_case": "web_application"},
]

import pandas as pd

df_ecs = pd.DataFrame(ecs_examples)
df_ebs = pd.DataFrame(ebs_examples)
df = pd.concat([df, df_ecs, df_ebs]).reset_index(drop=True)

print(f"\n✅ Added {len(ecs_examples)} ECS examples")
print(f"✅ Added {len(ebs_examples)} EBS examples")

# Counter-examples to reduce bias
counter_examples = [
    {"text": "Host a simple HTML/CSS/JS website", "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    {"text": "Deploy a basic landing page with no backend", "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    {"text": "Serve static files for documentation", "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    {"text": "Store backup files securely in cloud", "labels": "S3,IAM", "use_case": "data_processing"},
    {"text": "Archive old documents and files", "labels": "S3,IAM", "use_case": "data_processing"},
    {"text": "Upload and store media files", "labels": "S3,IAM", "use_case": "storage"},
    {"text": "Send push notifications to mobile users", "labels": "SNS,IAM", "use_case": "messaging"},
    {"text": "Queue background jobs for processing", "labels": "SQS,IAM", "use_case": "messaging"},
    {"text": "Build a serverless web application with user login", "labels": "Lambda,API_Gateway,DynamoDB,Cognito,IAM,CloudFront", "use_case": "serverless"},
    {"text": "Create a web API without managing servers", "labels": "Lambda,API_Gateway,IAM,DynamoDB", "use_case": "serverless"},
    {"text": "Store user data without provisioning servers", "labels": "DynamoDB,IAM", "use_case": "database"},
    {"text": "Build a traditional REST API on virtual machines", "labels": "EC2,VPC,IAM,RDS,API_Gateway", "use_case": "api_service"},
    {"text": "Deliver content globally with low latency", "labels": "CloudFront,S3,IAM", "use_case": "static_website"},
    {"text": "Implement user signup and login system", "labels": "Cognito,IAM", "use_case": "authentication_system"},
    {"text": "Process streaming events in real-time", "labels": "Lambda,IAM,SQS", "use_case": "data_processing"},
]

df_counter = pd.DataFrame(counter_examples)
df = pd.concat([df, df_counter]).reset_index(drop=True)

print(f"✅ Added {len(counter_examples)} counter-examples")

# Shuffle entire dataset
df = df.sample(frac=1).reset_index(drop=True)
all_labels_list = []
for labels_str in df['labels']:
    all_labels_list.extend(labels_str.split(','))

label_counts = Counter(all_labels_list)

print("\n📊 Label distribution after data generation:")
for service, count in label_counts.most_common():
    print(f"{service}: {count}")

# Save to CSV
df.to_csv('data/dataset_day2.csv', index=False)
print(f"✅ Generated {len(df)} examples")
