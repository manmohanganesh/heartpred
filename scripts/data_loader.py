import pandas as pd
def load_data():
    path='data/heart.csv'
    try:
        df=pd.read_csv(path)
        print("Data loaded successfully!")
        return df
    
    except FileNotFoundError:
        print(f"File not found at: {path}") 