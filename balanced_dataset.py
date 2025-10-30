import pandas as pd
from collections import Counter

df = pd.read_csv('data/dataset_day2_final.csv')

# Check current distribution
all_labels = []
for labels_str in df['labels']:
    all_labels.extend(labels_str.split(','))

label_counts = Counter(all_labels)

print("Current distribution:")
for service, count in sorted(label_counts.items(), key=lambda x: x[1]):
    print(f"{service}: {count}")

# Find underrepresented services (count < 20)
underrepresented = [svc for svc, count in label_counts.items() if count < 20]

print(f"\n⚠️ Underrepresented services: {underrepresented}")

# Manually add examples for underrepresented services
# Example:
if "EBS" in underrepresented:
    ebs_examples = [
        {"text": "Attach persistent disk storage to EC2 instance",
         "labels": "EC2,EBS,VPC,IAM", "use_case": "compute"},
        {"text": "Need block storage for database server",
         "labels": "EC2,EBS,RDS,VPC,IAM", "use_case": "database"},
        # Add 5-10 more
    ]
    df = pd.concat([df, pd.DataFrame(ebs_examples)]).reset_index(drop=True)

# Save final balanced dataset
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/dataset_balanced.csv', index=False)

print(f"\n✅ Final dataset: {len(df)} examples")