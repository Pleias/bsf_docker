import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import lancedb
import time
from typing import Literal

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Define the device
device = "cpu"
model_name = "PleIAs/pleias_350m_rag"

# Get Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set the HF_TOKEN environment variable")

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="models")
model.to(device)

# Set tokenizer configuration
tokenizer.eos_token = "<|answer_end|>"
eos_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 1

# Define variables 
temperature = 0.0  
max_new_tokens = 1200
top_p = 0.95
repetition_penalty = 1.0
min_new_tokens = 300
early_stopping = False



def search(text, table=None):
    print("searching")
    start = time.time()
    results = table.search(text, query_type="fts").limit(5).to_pandas()
    
    # Add a check for duplicate hashes
    seen_hashes = set()
    
    document = []
    fiches_html = []
    for _, row in results.iterrows():
        hash_id = str(row['hash'])
        
        # Skip if we've already seen this hash
        if hash_id in seen_hashes:
            continue
            
        seen_hashes.add(hash_id)
        title = row['section']
        content = row['text']

        document.append(f"<|source_start|><|source_id_start|>{hash_id}<|source_id_end|>{title}\n{content}<|source_end|>")
        fiches_html.append(f'<div class="source" id="{hash_id}"><p><b>{hash_id}</b> : {title}<br>{content}</div>')

    document = "\n".join(document)
    fiches_html = '<div id="source_listing">' + "".join(fiches_html) + "</div>"
    print(f"FTS search time: {time.time() - start:.2f} seconds")
    return document, fiches_html, results

class pleiasBot:
    def __init__(self, 
                 table_name: Literal["fr_fts", "en_fts", "both_fts"] = "both_fts", 
                 system_prompt="Tu es Appli, un assistant de recherche qui donne des réponses sourcées"
                 ):
        
        self.system_prompt = system_prompt
        # Connect to the LanceDB database
        db = lancedb.connect(f"data/{table_name}")
        self.table = db.open_table("crsv")

    def predict(self, user_message):
        fiches, fiches_html, search_results = search(user_message, table=self.table)
        
        detailed_prompt = f"""<|query_start|>{user_message}<|query_end|>\n{fiches}\n<|source_analysis_start|>"""

        # Convert inputs to tensor
        start = time.time()
        input_ids = tokenizer.encode(detailed_prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        try:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True, # replace False to avoid warning
                early_stopping=early_stopping,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode the generated text
            generated_text = tokenizer.decode(output[0][len(input_ids[0]):])
            print(f"Generation time: {time.time() - start:.2f} seconds")
 
            
            return generated_text, fiches_html, search_results

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None


# Launch the app
if __name__ == "__main__":
    pleias_bot = pleiasBot(table_name="fr_fts")
    results = pleias_bot.predict("responsabilité des Etats")
    print(results[2].to_dict())