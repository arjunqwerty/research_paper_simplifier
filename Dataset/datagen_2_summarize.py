import pandas as pd
import time
from huggingface_hub import InferenceClient
import re

# List of Hugging Face tokens (replace with your actual tokens)
HF_TOKENS = []

# Parquet file
output_file = "research_papers_extended.parquet"
current_token_index = 6
chunkSize = 1000
finished = False

# Function to clean text
def clean_text(text):
    # Remove unwanted symbols, extra spaces, and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\[\d+\]', ' ', text)  # Remove citation markers like [1], [2], etc.
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'/uni[0-9a-fA-F]+', ' ', text)
    # text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)  # Remove special characters except basic punctuation
    return text.strip()

# Function to chunk text
def chunk_text(text, max_tokens=1000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:  # Ensure the chunk is not empty
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:  # Add the last chunk if it's not empty
        chunks.append('. '.join(current_chunk) + '.')

    # Filter out empty or very short chunks
    return chunks

# Function to summarize text
def summarize_text(text, max_summary_words):
    global current_token_index
    prompt = f"Summarize this part of the research paper to less than {max_summary_words} words:\n{text}"
    retries = len(HF_TOKENS)
    tok_used = 0
    for attempt in range(len(HF_TOKENS)):
        try:
            client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKENS[current_token_index])
            return client.text_generation(prompt, max_new_tokens=200).strip()
        except Exception as e:
            if "429" in str(e) or "402" in str(e):  # Rate limit exceeded
                tok_used += 1
                if tok_used==retries:
                    return "DONE"
                # print(f"Rate limit exceeded for token {current_token_index}, switching to next token...\nConsecutive retries: {tok_used}")
                print(f"Error: {e}\n for token {current_token_index}, switching to next token...\nConsecutive retries: {tok_used}")
                current_token_index = (current_token_index + 1) % len(HF_TOKENS)  # Cycle to the next token
            else:
                print(f"Error: {e}, retrying ({attempt+1}/{retries})...")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
    return None

# Load dataset
df = pd.read_parquet(output_file)

# Generate summaries for missing ones
for idx, row in df.iterrows():
    if pd.isna(row["summary"]) and isinstance(row["input"], str):
        print(f"{idx}: Summarizing: {row['title']}")
        input_text = row["input"]
        cleaned_text = clean_text(input_text)
        print(f"Cleaned text from {len(input_text.split())} to {len(cleaned_text.split())}")
        chunks = chunk_text(cleaned_text, chunkSize)
        n = len(chunks)
        chunk_summaries = []

        for j, chunk in enumerate(chunks):
            print(f"->\tSummarizing chunk {j+1}/{n} of length {len(chunk.split())}...")
            chunk_summary = summarize_text(chunk, len(chunk.split())//10)
            if chunk_summary=="DONE":
                finished=True
                break
            if chunk_summary:
                chunk_summaries.append(chunk_summary)
        if finished: break
        summary = " ".join(chunk_summaries)

        if summary:
            df.at[idx, "summary"] = summary
df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")

if finished:
    print("ALL TOKENS EXHAUSTED")

print("Summarization completed.")
