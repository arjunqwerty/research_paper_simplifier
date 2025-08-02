import streamlit as st
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PyPDF2 import PdfReader
import re
import nltk
import time
import gc

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# === Setup ===
model_name = "model/ProphetNet"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Utility Functions ===
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    return text.strip()

def chunk_text(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, tokenizer, model):
    n = len(text.split()) // 10
    prompt = f"Summarize this part of the research paper to less than {n} words:\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=n+100, num_return_sequences=1)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_story(summary, tokenizer, model):
    story_prompt = f"""You are a master storyteller, known for crafting immersive and emotionally engaging stories that captivate readers of all ages.
Below is a "summary" of a research paper. Your task is to transform this summary into a "fully developed short story" with a natural flow, engaging characters, and a compelling plot.
## Guidelines for the Story:
- **Creative & Narrative-Driven**: Do not sound like a research paper. The story should feel "organic, engaging, and immersive".
- **Well-Developed Characters**: Introduce "relatable, human-like" characters with clear motivations.
- **Flow & Pacing**: The story should "unfold naturally" with a clear "beginning, middle, and end".
- **Easily Understandable**: Use "simple, conversational, yet elegant language" that anyone can enjoy.
- **Show, Don't Tell**: Use "vivid descriptions" and "natural dialogue" instead of just explaining ideas.
- **Engaging Conflict & Resolution**: The story should have "a central conflict" that gets resolved meaningfully.
---
### **Summary:**
{summary}
---
### Now, weave this into a captivating short story.
Ensure it feels like a "real, immersive narrative", not an AI-generated text. Make it flow like a professionally written short story. End the story naturally with a satisfying conclusion.
(Stop generating as soon as the story is complete.)"""
    max_tokens = min(1024, tokenizer.model_max_length - 20)
    inputs = tokenizer(story_prompt, return_tensors="pt", truncation=False, max_length=max_tokens).to(device)
    story_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return tokenizer.decode(story_ids[0], skip_special_tokens=True)

# === UI CONFIGURATION ===
st.set_page_config(page_title="Scientific Paper Simplifier", layout="centered")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        # .main {
        #     background-color: #f8f9fa;
        # }
        .title {
            font-size: 2.5rem !important;
            font-weight: 700;
        }
        .subtitle {
            font-size: 1.2rem !important;
            color: #6c757d;
        }
        .section-title {
            font-size: 1.4rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #333;
        }
        .stSpinner > div > div {
            color: #0d6efd;
        }
        .stButton > button {
            border-radius: 10px;
            background-color: #0d6efd;
            color: white;
            font-weight: 500;
            padding: 0.6em 1.2em;
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=80)
st.sidebar.markdown("## 🔍 How it Works")
st.sidebar.markdown("""
1. Upload a research paper in PDF  
2. Click **Submit**  
3. Get an engaging story from the content  
""")
st.sidebar.markdown("---")
st.sidebar.caption("Crafted with ❤️ for Capstone Excellence")

# === Main Title ===
st.markdown('<p class="title">📚 Paper2Story</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform research papers into captivating stories using AI</p>', unsafe_allow_html=True)

# === Upload Section ===
uploaded_file = st.file_uploader("📄 Upload your research paper (PDF)", type=["pdf"])
submit_clicked = st.button("🚀 Submit and Generate Story")

if uploaded_file and submit_clicked:
    st.markdown('<div class="section-title">1️⃣ Extracting & Preprocessing</div>', unsafe_allow_html=True)
    with st.spinner("Reading and cleaning your document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)

    st.markdown('<div class="section-title">2️⃣ Summarization in Progress</div>', unsafe_allow_html=True)
    with st.spinner("Loading Model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    summaries = []
    progress = st.progress(0, text="Summarizing content...")
    for i, chunk in enumerate(chunks):
        summaries.append(summarize_text(chunk, tokenizer, model))
        progress.progress((i + 1) / len(chunks))

    combined_summary = " ".join(summaries)

    with st.spinner("Generating the story..."):
        story = generate_story(combined_summary, tokenizer, model)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    st.success("✅ Story Generated Successfully!")
    st.markdown('<div class="section-title">📖 Final Output</div>', unsafe_allow_html=True)

    formatted_story = ".\n".join(story.split(". "))
    st.text_area("Your Story", formatted_story, height=400)

    st.download_button("💾 Download as .txt", story, file_name="paper2story_output.txt", use_container_width=True)

elif uploaded_file and not submit_clicked:
    st.info("Click the Submit button to begin story generation.")
