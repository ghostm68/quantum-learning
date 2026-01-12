import pandas as pd
import os

print("Files in current directory:")
print(os.listdir('.'))

try:
    df = pd.read_csv('inkrealm_clean.csv')
    print(f"\n✅ File loaded successfully! Shape: {df.shape}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumn names:")
    print(df.columns.tolist())
except FileNotFoundError:
    print("\n❌ File 'inkrealm_clean.csv' not found. Please ensure it's in the repository.")
except pd.errors.EmptyDataError:
    print("\n❌ The file is empty.")
except Exception as e:
    print(f"\n❌ An error occurred: {e}")
