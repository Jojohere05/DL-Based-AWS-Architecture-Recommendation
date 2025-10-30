import pandas as pd
from collections import Counter

df = pd.read_csv('data/dataset_day1.csv')

print(f"Total examples: {len(df)}")
print(f"\nSample examples:")
print(df.head(10))

# Check label distribution
all_labels = []
for labels_str in df['labels']:
    all_labels.extend(labels_str.split(','))

label_counts = Counter(all_labels)

print("\n📊 Service frequency:")
for service, count in sorted(label_counts.items()):
    print(f"{service}: {count}")

# Check for services with <10 examples (too few)
print("\n⚠️ Services with <10 examples:")
for service, count in label_counts.items():
    if count < 10:
        print(f"{service}: {count} ← Need more!")