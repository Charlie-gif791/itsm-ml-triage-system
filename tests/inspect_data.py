import pandas as pd

# Inspect issues.csv
issues = pd.read_csv("data/raw/issues.csv")

for col in issues.columns:
    print(col, "— unique values:", issues[col].nunique())

print()
# Inspect sample_utterances.csv
utterances = pd.read_csv("data/raw/sample_utterances.csv")

for col in utterances.columns:
    print(col, "— example:", utterances[col].iloc[0])
