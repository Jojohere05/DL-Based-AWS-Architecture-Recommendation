import pandas as pd

df = pd.read_csv('data/dataset_day2.csv')

# Negative examples (simple use cases that DON'T need complex services)
negative_examples = [
    # Static websites - should NOT have EC2, RDS, Lambda
    {"text": "Host a simple portfolio website", 
     "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    
    {"text": "Deploy a static landing page", 
     "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    
    {"text": "Create a documentation site", 
     "labels": "S3,CloudFront,IAM", "use_case": "static_website"},
    
    # Simple storage - should NOT have compute
    {"text": "Store backup files securely", 
     "labels": "S3,IAM", "use_case": "data_processing"},
    
    {"text": "Archive old documents", 
     "labels": "S3,IAM", "use_case": "data_processing"},
    
    # Notification only - should NOT have database or compute
    {"text": "Send push notifications to users", 
     "labels": "SNS,IAM", "use_case": "messaging"},
    
    {"text": "Alert users about system events", 
     "labels": "SNS,Lambda,IAM", "use_case": "messaging"},
    
    # Add 20-30 more negative examples
    {"text": "Queue messages for processing later",
     "labels": "SQS,IAM", "use_case": "messaging"},
     
    {"text": "Deliver content globally with low latency",
     "labels": "CloudFront,S3,IAM", "use_case": "static_website"},
     
    {"text": "Store JSON configuration files",
     "labels": "S3,IAM", "use_case": "storage"},
]

df_negative = pd.DataFrame(negative_examples)
df = pd.concat([df, df_negative]).reset_index(drop=True)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('data/dataset_day2_final.csv', index=False)
print(f"✅ Total examples: {len(df)}")