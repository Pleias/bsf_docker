import lancedb
import time
from typing import Literal
from src.generation import GenerationEngine
import json

import logging

logger = logging.getLogger(__name__)


class PleiasBot:
    def __init__(self, 
                 table_name: Literal["fr_fts", "en_fts", "both_fts"] = "both_fts", 
                 model_path = "models/Pleias-RAG-350m.gguf",
                 temperature = 0.0,
                 max_new_tokens = 2048,
                 top_p = 0.95,
                 repetition_penalty = 1.0,
                 search_limit = 3,
                 ):
        
        self.generation_engine = GenerationEngine(
                model_path_or_name=model_path,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                backend="llama_cpp")
        # Connect to the LanceDB database
        db = lancedb.connect(f"data/{table_name}")
        self.table = db.open_table("crsv")
        self.search_limit = search_limit
        
    def search(self, text, table=None, limit=3):
        
        # Search table
        logger.info("Searching for text")
        start = time.time()
        results = table.search(text, query_type="fts").limit(limit).to_pandas().T.to_dict() ### REPLACE BY 5
        logger.info(f"Search time: {time.time() - start:.2f} seconds")
        
        # Reformat the results to match the expected structure
        sources = []
        for idx, key in enumerate(results.keys(), 1):
            sources.append({
                "id": idx,
                "text": results[key]["text"],
                "metadata": {
                    subkey: results[key][subkey] for subkey in results[key].keys() if subkey != "text"
                }
            })
            
        return sources

    def predict(self, user_message):
        sources = self.search(user_message, table=self.table, limit=self.search_limit)
        start = time.time()
        logger.info("Generating response")

        try:
            response = self.generation_engine.generate(
                query=user_message,
                sources=sources,
            )
            logger.info(f"Total time: {time.time() - start:.2f} seconds")
            logger.debug(f"Response:\n{json.dumps(response, indent=2, ensure_ascii=False)}")
            return response

        except Exception as e:
            logger.info(f"Error during generation: {str(e)}")
            import traceback
            traceback.logger.info_exc()
            return None
        
    def stream_predict(self, user_message: str):
        """
        Yields raw text pieces from the model, _including_ all reasoning
        up through the <|answer_end|> tag.
        """
        # 1) run your search so we have sources in-hand
        sources = self.search(user_message, table=self.table, limit=self.search_limit)
        prompt = self.generation_engine.format_prompt(user_message, sources)
        
        # 2) generate token‐by‐token
        for piece in self.generation_engine._generate_llama_cpp_stream(prompt):
            print(piece, end="", flush=True)  # Print the raw text piece
            yield piece
            
        # 3) after the stream finishes, return the final processed sections
        raw = self.generation_engine._last_raw  
        sections = self.generation_engine.extract_sections(raw)
        if "answer" in sections:
            info = self.generation_engine.extract_citations(sections["answer"], sources)
            sections["clean_answer"] = info["clean_text"]
            sections["citations"]   = info["citations"]
        yield {"__done__": sections, "sources": sources}

# Launch the app
if __name__ == "__main__":
    pleias_bot = PleiasBot(table_name="fr_fts")
    results = pleias_bot.predict("Quelle est la responsabilité des Etats en matière de violences sexuelles ?")
    logger.info(results)
