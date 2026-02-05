import pandas as pd
from sklearn.model_selection import train_test_split

def make_train_val_splits(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Returns:
        train_df, val_df
    """

    df = df[[text_col, label_col]].copy()

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    return train_df, val_df
