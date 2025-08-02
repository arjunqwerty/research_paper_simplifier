import arxiv
import requests
import pandas as pd
import os
import time
from PyPDF2 import PdfReader
import re

# Output folder
output_folder = "downloaded_papers"
os.makedirs(output_folder, exist_ok=True)

# Parquet file for saving results incrementally
output_file = "research_papers_extended.parquet"

# Define research domains
domains = ["astronomy", "environmental sciences", "health AND medicine", "artificial intelligence", "robotics", "space exploration"]
papers_per_domain = 50
num_paper_limit = 50
save_every = 10
start_index = int(input("Enter start index: "))

# Function to download PDFs
def download_pdf(url, paper_id, retries=5):
    pdf_path = os.path.join(output_folder, f"{paper_id}.pdf")
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(pdf_path, "wb") as file:
                    file.write(response.content)
                return pdf_path
            else:
                print(f"Failed to download {url}. Status code: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print(f"Timeout while downloading {url}. Retrying {attempt + 1}/{retries}...")
            time.sleep(10)  # Wait before retrying

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}. Retrying {attempt + 1}/{retries}...")
            time.sleep(60)
    print(f"Failed to download {url} after {retries} attempts. Skipping...")
    return None

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    return text.strip()

# Function to extract text
def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return clean_text(text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

# # Load existing dataset
# if os.path.exists(output_file):
#     df = pd.read_parquet(output_file)
# else:
#     df = pd.DataFrame(columns=["title", "domain", "input", "summary", "story"])

'''# Fetch research papers
for domain in domains:
    search = arxiv.Search(query=domain, max_results=papers_per_domain + start_index, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    for i, result in enumerate(arxiv.Client().results(search)):
        if i < start_index:
            continue
        if not df[df["title"] == result.title].empty:
            continue

        print(f"Processing: {result.title}")
        paper_id = result.entry_id.split("/")[-1]
        pdf_path = download_pdf(result.pdf_url, paper_id)
        if not pdf_path:
            continue

        text = extract_text(pdf_path)
        if not text:
            continue

        # Append new data to DataFrame
        new_data = pd.DataFrame([{"title": result.title, "domain": domain, "input": text, "summary": None, "story": None}])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
'''

# Fetch research papers
cnt = 0
for domain in domains:
    # start_index = 0  # Reset for each domain
    papers_fetched = 0
    df = pd.read_parquet(output_file)
    start_index1 = start_index
    print(f"---> Starting {domain}...")
    search = arxiv.Search(
        query=domain,
        max_results=start_index1 + (2*num_paper_limit),  # Reduce to avoid empty results
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    print(f"Fetched {start_index1+(2*num_paper_limit)} papers with {start_index1} index")
    papers_fetched = 0  # Track successful downloads
    for i, result in enumerate(arxiv.Client().results(search)):
        if i<start_index1: continue
        if not df[df["title"] == result.title].empty: continue
        if papers_fetched >= num_paper_limit: break

        cnt += 1
        print(f"{cnt}. Processing: {result.title}")
        # paper_id = result.entry_id.split("/")[-1]
        paper_id = os.path.join(domain, result.entry_id.split("/")[-1])
        if not os.path.exists(os.path.join(output_folder, f"{paper_id}.pdf")):
            pdf_path = download_pdf(result.pdf_url, paper_id)
            if not pdf_path:
                print("Downloading pdf not done properly. Skipping extraction...")
                continue

        text = extract_text(pdf_path)
        if not text:
            print("Extracted text is empty. Skipping concatenation...")
            continue

        new_data = pd.DataFrame([{"title": result.title, "domain": domain, "input": text, "summary": None, "story": None}])
        df = pd.concat([df, new_data], ignore_index=True)
        papers_fetched += 1
        print("Concatenated the text to dataframe")
        if papers_fetched % save_every==0:
            df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
            print(f"Conctenated {save_every} papers and saved in parquet file.")

    if papers_fetched == 0:
        print(f"No more papers found for {domain}. Stopping early.")
        break  # Stop if no new papers were added
        
    print(f"Downloaded and extracted {papers_fetched} papers in {domain} domain")
    # start_index1 += 50  # Increment start index in small steps
    df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
    print(f"Conctenated {domain} papers and saved in parquet file.")

print("Download and extraction completed.")
