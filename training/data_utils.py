import pandas as pd

def rebalance_training_data(
        train_df: pd.DataFrame, 
        label_col: str = "label", 
        imbalance_ratio: int = 3 # majority can be at most 3x minority
    ) -> pd.DataFrame:
    ticket_df = train_df[train_df[label_col] == "Ticket"]
    dep_df = train_df[train_df[label_col] == "Deployment"]
    hd_df = train_df[train_df[label_col] == "HD Service"]

    minority_max = max(len(dep_df), len(hd_df))
    ticket_cap = minority_max * imbalance_ratio

    ticket_df = ticket_df.sample(
        n=min(ticket_cap, len(ticket_df)),
        random_state=42
    )

    balanced_df = pd.concat([ticket_df, dep_df, hd_df])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)