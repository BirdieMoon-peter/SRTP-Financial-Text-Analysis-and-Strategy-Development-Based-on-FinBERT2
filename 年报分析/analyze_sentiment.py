"""
Author: Peter Li
Date: 2025-11-13 14:55:56
Description: 调用finbert模型，计算句子的embedding
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import re
from tqdm import tqdm

def get_document_embedding(sentences: list, model, tokenizer, device, batch_size=32):
    """
    Calculates the mean CLS embedding for a list of sentences.
    """
    if not sentences:
        return None
    
    all_cls_embeddings = []
    # Use tqdm for progress bar on sentence batches, but keep it unobtrusive
    for i in tqdm(range(0, len(sentences), batch_size), desc="  - Sentences", leave=False, ncols=80):
        batch_sentences = sentences[i:i+batch_size]
        
        inputs = tokenizer(
            batch_sentences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_cls_embeddings.append(cls_embeddings)

    if not all_cls_embeddings:
        return None
        
    all_cls_embeddings_tensor = torch.cat(all_cls_embeddings, dim=0)
    document_embedding = torch.mean(all_cls_embeddings_tensor, dim=0)
    
    return document_embedding.numpy()

def main():
    """
    Main function to process CSVs and generate embeddings.
    """
    # --- 1. Setup ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv_dir = os.path.join(base_dir, 'csv_output')
    output_file = os.path.join(base_dir, 'finbert_embeddings.csv')

    if not os.path.exists(input_csv_dir):
        print(f"Error: Input directory not found at '{input_csv_dir}'")
        print("Please run the 'process_reports.py' script first to generate the sentence CSVs.")
        return

    # --- 2. Load Model ---
    print("Loading FinBERT model and tokenizer (valuesimplex-ai-lab/FinBERT2-base)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained("valuesimplex-ai-lab/FinBERT2-base")
        model = AutoModel.from_pretrained("valuesimplex-ai-lab/FinBERT2-base").to(device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPlease ensure you have an internet connection and required libraries.")
        print("You can install them with: pip install transformers torch pandas tqdm")
        return

    # --- 3. Process Files ---
    all_results = []
    csv_files = sorted([f for f in os.listdir(input_csv_dir) if f.endswith('.csv')])
    
    print(f"\nFound {len(csv_files)} CSV files to process in '{input_csv_dir}'.")

    # Use tqdm for progress bar on files
    for filename in tqdm(csv_files, desc="Processing Files", ncols=100):
        # Parse filename: 2018_000001_平安银行.csv
        match = re.match(r'(\d{4})_(\w+)_([\w\W]+).csv', filename)
        if not match:
            tqdm.write(f"Warning: Skipping file with unexpected name format: {filename}")
            continue
        
        year, company_code, company_name = match.groups()
        
        file_path = os.path.join(input_csv_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if 'sentence' not in df.columns or df['sentence'].isnull().all():
                tqdm.write(f"Warning: No sentences found in {filename}. Skipping.")
                continue

            sentences = df['sentence'].dropna().tolist()
            
            # Get the average embedding for the whole document
            doc_embedding = get_document_embedding(sentences, model, tokenizer, device)

            if doc_embedding is not None:
                result = {
                    'year': year,
                    'company_code': company_code,
                    'company_name': company_name,
                }
                # Add embedding vectors to the result dictionary
                for i, val in enumerate(doc_embedding):
                    result[f'vec_{i}'] = val
                all_results.append(result)

        except Exception as e:
            tqdm.write(f"Error processing file {filename}: {e}")

    # --- 4. Save Results ---
    if not all_results:
        print("\nNo embeddings were generated. Exiting.")
        return

    print("\nSaving aggregated embeddings to CSV...")
    final_df = pd.DataFrame(all_results)
    
    # Dynamically get the number of vector columns from the last processed embedding
    if 'doc_embedding' in locals() and doc_embedding is not None:
        vec_cols = [f'vec_{i}' for i in range(len(doc_embedding))]
        info_cols = ['year', 'company_code', 'company_name']
        final_df = final_df[info_cols + vec_cols]

    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Successfully saved results to {output_file}")


if __name__ == "__main__":
    main()
