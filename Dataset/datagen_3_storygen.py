import pandas as pd
import time
from huggingface_hub import InferenceClient

# Parquet file
output_file = "research_papers_extended.parquet"

# List of Hugging Face tokens
HF_TOKENS = []
current_token_index = 6
story=""

def generate_text(text, max_words):
    global current_token_index
    # prompt = f"""Using the following research summary, generate an engaging story of {max_words} words. Ensure the story has: 1. Use only the researchers, their roles, or the technology being developed as characters. Do not introduce random names. 2. The setting should reflect the research field 3. The conflict or problem should represent the problem addressed in the research. 4. A resolution should reflect the findings. 5. Ensure an engaging, easy-to-understand narrative style. Research Summary: {text}"""
    # prompt = f"""Create an engaging story based on the following research paper of {max_words} words. The story should seamlessly incorporate the key elements of the research in a narrative format. The characters should represent the researchers, the subject of study, or the technology being developed—avoid using arbitrary names. The setting should reflect the environment where the research takes place. The story should highlight the main problem the research aims to solve, the challenges faced, and the breakthrough that led to the solution. Clearly explain the method used to arrive at the solution in a way that is engaging and understandable. Ensure the story is immersive and flows naturally while staying true to the research content. Stop immediately after 1000 words. \n\nResearch Paper: {text}"""
    prompt = f"""You are a master storyteller, known for crafting immersive and emotionally engaging stories that captivate readers of all ages.
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
{text}
---
### Now, weave this into a captivating short story.
Ensure it feels like a "real, immersive narrative", not an AI-generated text. Make it flow like a professionally written short story. End the story naturally with a satisfying conclusion.
(Stop generating as soon as the story is complete.)
"""
    retries = len(HF_TOKENS)
    tok_used = 0
    for attempt in range(retries):
        try:
            client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKENS[current_token_index])
            summary = client.text_generation(prompt, max_new_tokens=max_words, temperature=0.7, repetition_penalty=1.1)
            return summary.strip()
        except Exception as e:
            if "429" in str(e) or "402" in str(e):  # Rate limit exceeded
                tok_used += 1
                if tok_used==retries:
                    return "DONE"
                print(f"Error: {e}\n for token {current_token_index}, switching to next token...\nConsecutive retries: {tok_used}")
                # print(f"Rate limit exceeded for token {current_token_index}, switching to next token...\nConsecutive retries: {tok_used}")
                current_token_index = (current_token_index + 1) % len(HF_TOKENS)  # Cycle to the next token
            else:
                print(f"Error: {e}, retrying ({attempt+1}/{retries})...")
            time.sleep(5 * (attempt + 1))  # Exponential backoff
    return None

# Load dataset
df = pd.read_parquet(output_file)

# Generate stories for missing ones
for idx, row in df.iterrows():
    if pd.isna(row["story"]) and isinstance(row["summary"], str):
        print(f"{idx}: Generating story for: {row['title']}")
        story = generate_text(row["summary"], 1000)
        if story=="DONE": break
        if story:
            df.at[idx, "story"] = story
        df.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")
if story=="DONE":
    print("ALL TOKENS EXHAUSTED")

print("Story generation completed.")
