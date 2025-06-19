import pandas as pd

def load_data(filepath):
    """
    Load CSV data from the given path.
    """
    return pd.read_csv(filepath)
