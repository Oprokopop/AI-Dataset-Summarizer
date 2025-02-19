import pandas as pd
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_dataset(file_path):
    """Generates a summary for a dataset based on its descriptive statistics."""
    df = pd.read_csv(file_path)
    stats = df.describe().to_string()
    prompt = f"Summarize the following dataset statistics:\n{stats}\nProvide a concise overview of what these statistics indicate about the data."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    file_path = input("Enter the path to the CSV file: ")
    summary = summarize_dataset(file_path)
    print("Dataset Summary:\n", summary)
