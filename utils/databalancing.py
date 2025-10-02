import pandas as pd
import numpy as np

def rebalance_dataset(df, class_col='are_same', observed_cols=['name1', 'name2'], random_state=None):
    """
    Rebalances dataset by oversampling the minority class using synthetic pairs.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        class_col (str): Column defining the class
        observed_cols (list[str]): Columns used for synthetic sampling
        random_state (int or None): Seed for reproducibility
    
    Returns:
        pd.DataFrame: Rebalanced dataframe
    """
    rng = np.random.default_rng(random_state)
    
    # Determine class distribution
    class_counts = df[class_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    n_to_add = class_counts[majority_class] - class_counts[minority_class]
    
    if n_to_add == 0:
        return df.copy()
    
    # Extract values from observed columns
    col_values = {col: df[col].unique() for col in observed_cols}
    
    # Set of existing pairs to avoid duplicates
    existing_pairs = set(zip(df[observed_cols[0]], df[observed_cols[1]]))
    
    new_rows = []
    attempts = 0
    max_attempts = n_to_add * 10  # safeguard otherwise it keeps trying to make non-existing combos that are not possible any longer.
    
    while len(new_rows) < n_to_add and attempts < max_attempts:
        attempts += 1
        val1 = rng.choice(col_values[observed_cols[0]])
        val2 = rng.choice(col_values[observed_cols[1]])
        
        if val1 == val2:
            continue
        if (val1, val2) in existing_pairs or (val2, val1) in existing_pairs:
            continue
        
        new_rows.append({observed_cols[0]: val1,
                         observed_cols[1]: val2,
                         class_col: minority_class})
        
        existing_pairs.add((val1, val2))
    
    if len(new_rows) < n_to_add:
        print(f"Only generated {len(new_rows)} of {n_to_add} needed samples (limited combinations).")
    
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
