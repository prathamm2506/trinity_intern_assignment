# CSV Header Description Generator

This script uses the **GPT-2** language model (via Hugging Face Transformers) to generate short, descriptive text for CSV column headers. GPT-2 was chosen because it can run fully **offline**, requires minimal setup, and is capable of producing coherent text for small prompts, making it ideal for lightweight, local use without internet access.

## How to Run

1. Install the required dependencies:

```
pip install pandas transformers torch
```

2. Run the script with a CSV or text file containing headers:

```
python header_description_generator.py path/to/your_file.csv
```

If no file path is provided, the script will prompt for one. Descriptions are displayed in the console and saved to `output.txt` by default.

## Challenges

One challenge was **ensuring meaningful descriptions for short or ambiguous headers**, as GPT-2 is smaller and less context-aware than newer models. Another was handling **varied file formats** and missing data gracefully, which required careful error handling and fallback defaults.
