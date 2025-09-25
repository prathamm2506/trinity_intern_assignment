#!/usr/bin/env python3
"""
CSV Header Description Generator

This script reads CSV headers from a file and generates short descriptive text
for each header using an offline language model (GPT-2).

Requirements:
pip install pandas transformers torch

Author: Assistant
Date: 2025
"""

import csv
import os
import sys
from typing import List, Dict
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch


class HeaderDescriptionGenerator:
    """
    A class to generate descriptions for CSV headers using a local language model.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the generator with a specified model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load the language model for text generation."""
        try:
            print(f"Loading model: {self.model_name}...")
            # Create a text generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,  # Use CPU (-1) or GPU (0)
                torch_dtype=torch.float32
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def generate_description(self, header: str, max_length: int = 50) -> str:
        """
        Generate a descriptive text for a given header.
        
        Args:
            header (str): The CSV header to describe
            max_length (int): Maximum length of generated text
            
        Returns:
            str: Generated description
        """
        # Create a prompt that encourages descriptive output
        prompt = f"The column '{header}' in a dataset contains"
        
        try:
            # Generate text
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 20,  # Reasonable length
                num_return_sequences=1,
                temperature=0.7,  # Some creativity but not too random
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract the generated part (remove the prompt)
            generated_text = result[0]['generated_text']
            description = generated_text[len(prompt):].strip()
            
            # Clean up the description
            description = self._clean_description(description)
            
            return description
            
        except Exception as e:
            print(f"Error generating description for '{header}': {e}")
            return "data values"  # Fallback description
    
    def _clean_description(self, description: str) -> str:
        """
        Clean and format the generated description.
        
        Args:
            description (str): Raw generated description
            
        Returns:
            str: Cleaned description
        """
        # Remove extra whitespace
        description = description.strip()
        
        # Take only the first sentence or up to a reasonable length
        sentences = description.split('.')
        if sentences:
            description = sentences[0].strip()
        
        # Limit length
        if len(description) > 100:
            description = description[:100].strip()
        
        # Ensure it ends properly
        if not description.endswith('.'):
            description += '.'
        
        return description


class CSVHeaderReader:
    """
    A class to handle reading CSV headers from files.
    """
    
    @staticmethod
    def read_headers_from_csv(file_path: str) -> List[str]:
        """
        Read headers from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            List[str]: List of column headers
        """
        try:
            # Read only the first row to get headers
            df = pd.read_csv(file_path, nrows=0)
            headers = df.columns.tolist()
            
            # Clean headers (strip whitespace)
            headers = [header.strip() for header in headers]
            
            return headers
            
        except Exception as e:
            print(f"Error reading CSV file '{file_path}': {e}")
            return []
    
    @staticmethod
    def read_headers_from_text(file_path: str) -> List[str]:
        """
        Read headers from a text file (one header per line).
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            List[str]: List of headers
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                headers = [line.strip() for line in f if line.strip()]
            return headers
            
        except Exception as e:
            print(f"Error reading text file '{file_path}': {e}")
            return []


class OutputWriter:
    """
    A class to handle output writing to console and file.
    """
    
    @staticmethod
    def write_results(results: Dict[str, str], output_file: str = "output.txt"):
        """
        Write results to console and file.
        
        Args:
            results (Dict[str, str]): Dictionary of header -> description
            output_file (str): Output file name
        """
        # Write to console
        print("\n" + "="*60)
        print("HEADER DESCRIPTIONS")
        print("="*60)
        
        for header, description in results.items():
            print(f"\nHeader: {header}")
            print(f"Description: {description}")
        
        # Write to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("CSV Header Descriptions\n")
                f.write("=" * 50 + "\n\n")
                
                for header, description in results.items():
                    f.write(f"Header: {header}\n")
                    f.write(f"Description: {description}\n\n")
            
            print(f"\nResults also saved to: {output_file}")
            
        except Exception as e:
            print(f"Error writing to file '{output_file}': {e}")


def main():
    """
    Main function to orchestrate the header description generation process.
    """
    print("CSV Header Description Generator")
    print("=" * 40)
    
    # Get input file from user
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter path to CSV file (or text file with headers): ").strip()
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    # Determine file type and read headers
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if file_extension == '.csv':
        headers = CSVHeaderReader.read_headers_from_csv(input_file)
    else:
        headers = CSVHeaderReader.read_headers_from_text(input_file)
    
    if not headers:
        print("No headers found or error reading file.")
        sys.exit(1)
    
    print(f"\nFound {len(headers)} headers:")
    for i, header in enumerate(headers, 1):
        print(f"  {i}. {header}")
    
    # Initialize the description generator
    generator = HeaderDescriptionGenerator()
    
    # Generate descriptions
    print("\nGenerating descriptions...")
    results = {}
    
    for i, header in enumerate(headers, 1):
        print(f"Processing header {i}/{len(headers)}: {header}")
        description = generator.generate_description(header)
        results[header] = description
    
    # Write results
    OutputWriter.write_results(results)


if __name__ == "__main__":
    main()