import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the dataset
df = pd.read_csv("Roman-Urdu-Poetry.csv")

# Check the column names
if "Poetry" not in df.columns:
    raise ValueError(f"Expected 'Poetry' column, but found {df.columns}. Check column names.")

poetry_texts = df["Poetry"].astype(str).tolist()  # Use correct column name

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_texts)

# Save the tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved as tokenizer.pkl ✅")
