"""
Author: Peter Li
Date: 2025-11-13 14:56:46
Description: 从pdf文件中提取文本，清洗文本，将文本拆分成句子，保存到csv文件中
"""

import pdfplumber
import re
import pandas as pd
import os

def process_pdf(pdf_path, output_dir):
    """
    Reads a PDF file, extracts text, cleans it, splits it into sentences,
    and saves the sentences to a CSV file.
    """
    try:
        # --- 1. Extract information from the file path ---
        dir_name = os.path.basename(os.path.dirname(pdf_path))
        year_match = re.search(r'(\d{4})', dir_name)
        if not year_match:
            print(f"Warning: Could not extract year from directory '{dir_name}'. Skipping {pdf_path}")
            return

        year = year_match.group(1)

        file_name = os.path.basename(pdf_path)
        name_parts = file_name.replace('.pdf', '').split('_')
        if len(name_parts) < 2:
            print(f"Warning: Filename '{file_name}' does not match expected format '<code>_<name>_<year>.pdf'. Skipping.")
            return

        company_code = name_parts[0]
        company_name = name_parts[1]

        # --- 2. Define output path and check if it exists ---
        csv_filename = f"{year}_{company_code}_{company_name}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        if os.path.exists(csv_filepath):
            return

        # --- 3. Read PDF and extract text ---
        print(f"Processing: {pdf_path}")
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
        
        if not full_text:
            print(f"Warning: No text could be extracted from {pdf_path}.")
            return

        # --- 4. Clean text and split into sentences ---
        text_content = "".join(full_text)
        # Remove spaces, newlines, and tabs
        cleaned_text = re.sub(r'[\s\r\n\t\u3000]+', '', text_content)
        # Split into sentences based on punctuation
        sentences = re.split(r'(?<=[。！？])', cleaned_text)
        # Remove any leading/trailing whitespace from sentences and filter out empty ones
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            print(f"Warning: Text was extracted, but no sentences were formed from {pdf_path}.")
            return

        # --- 5. Save to CSV ---
        df = pd.DataFrame(sentences, columns=['sentence'])
        df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        print(f"Successfully created: {csv_filepath}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")


def main():
    """
    Main function to find all PDFs and process them.
    """
    # Get the absolute path of the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_root_folder = os.path.join(base_dir, 'PDF文件')
    output_csv_dir = os.path.join(base_dir, 'csv_output')

    if not os.path.exists(pdf_root_folder):
        print(f"Error: PDF directory not found at '{pdf_root_folder}'")
        return

    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)
        print(f"Created output directory: {output_csv_dir}")

    # Walk through all subdirectories and find PDF files
    for dirpath, _, filenames in os.walk(pdf_root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(dirpath, filename)
                process_pdf(pdf_path, output_csv_dir)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
