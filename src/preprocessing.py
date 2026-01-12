import pandas as pd
import unidecode
import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_equipment(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df['clean_name'] = df['equipment_name'].apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_equipment("data/raw/equipment_catalog.csv", "data/processed/equipment_clean.csv")