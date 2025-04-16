from llama_cpp import Llama
import lancedb
import time
from typing import Literal

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("llama_cpp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def search(text, table=None):
    logger.info("Searching for text")
    start = time.time()
    results = table.search(text, query_type="fts").limit(1).to_pandas() ### REPLACE BY 5
    
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
    logger.info(f"FTS search time: {time.time() - start:.2f} seconds")
    return document, fiches_html, results

class pleiasBot:
    def __init__(self, 
                 table_name: Literal["fr_fts", "en_fts", "both_fts"] = "both_fts", 
                 system_prompt="Tu es Appli, un assistant de recherche qui donne des réponses sourcées",
                 model_path = "models/rag350v3-bf16.gguf",
                 temperature = 0.0,
                 max_new_tokens = 1200,
                 top_p = 0.95,
                 repetition_penalty = 1.0,
                 ):
        
        # SYSTEM PROMPT USELESS NOW
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        
        self.system_prompt = system_prompt
        # Connect to the LanceDB database
        db = lancedb.connect(f"data/{table_name}")
        self.table = db.open_table("crsv")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,            # 2x plus rapide quand on ne met que 3000
            n_gpu_layers=0,        # cpu only 
        )

    def predict_stream(self, user_message):
        document, fiches_html, search_results = search(user_message, table=self.table)
        
        detailed_prompt = f"""<|query_start|>{user_message}<|query_end|>\n{document}\n<|source_analysis_start|>"""

        # Convert inputs to tensor
        start = time.time()

        try:
            logger.info("Generating response")
            tokens = self.model.generate(
                self.model.tokenize(detailed_prompt.encode("utf-8"), special=True), 
                temp=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repetition_penalty,
                reset=True,
            )
            generated_text = ""
                                
            for i, t in enumerate(tokens):
                piece = self.model.detokenize([t], special=True).decode("utf-8", errors="replace")
                if (piece == "<|answer_end|>") | (i >= self.max_new_tokens):
                    break
                generated_text += piece
                yield piece  # Stream the generated text
                
            # logger.info(f"Generation time: {time.time() - start:.2f} seconds")
 

        except Exception as e:
            logger.info(f"Error during generation: {str(e)}")
            import traceback
            traceback.logger.info_exc()
            return None, None, None


# Launch the app
if __name__ == "__main__":
    pleias_bot = pleiasBot(table_name="fr_fts")
    results = pleias_bot.predict("Quelle est la responsabilité des Etats en matière de violences sexuelles ?")
    logger.info(results[2].to_dict())
    logger.info("*"*50)
    logger.info(results[0])