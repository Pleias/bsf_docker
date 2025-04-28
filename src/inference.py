import json
import logging
import time
from typing import Literal

import lancedb

from src.generation import GenerationEngine

logger = logging.getLogger(__name__)


class PleiasBot:
    def __init__(
        self,
        table_name: Literal["fr", "en", "both"] = "both",
        model_path: str = "models/Pleias-RAG-350m.gguf",
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        search_limit: int = 3,
    ):
        """
        Orchestrates the generation and the data retrieval with the specified parameters.
        Args:
            table_name (Literal["fr", "en", "both"], optional):
                The name of the table to connect to in the LanceDB database.
                "fr" for French, "en" for English, or "both" for both languages.
                Defaults to "both".
            model_path (str, optional):
                The file path to the model to be used for generation.
                Defaults to "models/Pleias-RAG-350m.gguf".
            temperature (float, optional):
                The sampling temperature for the generation engine.
                Defaults to 0.0.
            max_new_tokens (int, optional):
                The maximum number of new tokens to generate.
                Defaults to 2048.
            top_p (float, optional):
                The nucleus sampling probability for the generation engine.
                Defaults to 0.95.
            repetition_penalty (float, optional):
                The penalty for repeated tokens during generation.
                Defaults to 1.0.
            search_limit (int, optional):
                The maximum number of search results to retrieve from the database.
                Defaults to 3.
        Attributes:
            generation_engine (GenerationEngine):
                The engine used for text generation with the specified parameters.
            table (lancedb.Table):
                The LanceDB table object connected to the specified database.
            search_limit (int):
                The maximum number of search results to retrieve.
        """

        # Initialize the generation engine with the specified parameters
        self.generation_engine = GenerationEngine(
            model_path_or_name=model_path,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            backend="llama_cpp",
        )
        # Connect to the LanceDB database
        db = lancedb.connect(f"data/{table_name}")
        self.table = db.open_table("crsv")
        self.search_limit = search_limit

    def search(self, text: str, table: lancedb.table.Table, limit: int = 3):
        """
        Perform a search operation on the given table using full-text search (FTS)
        and return the results in a structured format.
        Args:
            text (str): The text query to search for.
            table (object): The table object to perform the search on.
                Must support a `search` method with `query_type="fts"`.
            limit (int, optional): The maximum number of search results to return.
                Defaults to 3.
        Returns:
            list: A list of dictionaries representing the search results. Each dictionary
            contains the following keys:
                - "id" (int): The result index (1-based).
                - "text" (str): The text content of the result.
                - "metadata" (dict): A dictionary of additional metadata for the result,
                  excluding the "text" field.
        """

        # Search table
        logger.info("Searching for text")
        start = time.time()
        results = (
            table.search(text, query_type="fts").limit(limit).to_pandas().T.to_dict()
        )
        logger.info(f"Search time: {time.time() - start:.2f} seconds")

        # Reformat the results to match the expected structure
        sources = []
        for idx, key in enumerate(results.keys(), 1):
            sources.append(
                {
                    "id": idx,
                    "text": results[key]["text"],
                    "metadata": {
                        subkey: results[key][subkey]
                        for subkey in results[key].keys()
                        if subkey != "text"
                    },
                }
            )

        return sources

    def predict(self, user_message: str):
        """
        Generates a response based on the given user message by searching for relevant sources
        and utilizing a generation engine.
        Args:
            user_message (str): The input message from the user for which a response is to be generated.
        Returns:
            dict or None: The generated response as a dictionary if successful, or None if an error occurs.
        Logs:
            - Logs the total time taken for the response generation.
            - Logs the generated response in debug mode.
            - Logs any errors encountered during the generation process.
        """

        sources = self.search(user_message, table=self.table, limit=self.search_limit)
        start = time.time()
        logger.info("Generating response")

        try:
            response = self.generation_engine.generate(
                query=user_message,
                sources=sources,
            )
            logger.info(f"Total time: {time.time() - start:.2f} seconds")
            logger.debug(
                f"Response:\n{json.dumps(response, indent=2, ensure_ascii=False)}"
            )
            return response

        except Exception as e:
            logger.info(f"Error during generation: {str(e)}")
            import traceback

            traceback.logger.info_exc()
            return None

    def stream_predict(self, user_message: str):
        """
        Same method as `predict`, but streams the generation token-by-token.
        This method generates text pieces token-by-token from the model, including
        all reasoning up to the <|answer_end|> tag. When this is done, it also processes the final
        output to extract relevant sections and citations.
        Args:
            user_message (str): The input message from the user to generate predictions.
        Yields:
            str: Raw text pieces generated by the model during streaming.
            dict: A dictionary containing the final processed sections and sources
                  after the stream finishes. The dictionary includes:
                  - "__done__": Processed sections with keys such as "clean_answer"
                    (the cleaned answer text) and "citations" (extracted citations).
                  - "sources": The sources retrieved during the search step.
        Notes:
            - The method first performs a search to retrieve relevant sources based
              on the user's message.
            - It formats a prompt using the retrieved sources and streams the
              generation token-by-token.
            - After streaming, it processes the raw output to extract sections
              and citations.
            - Debug logs can be enabled to print raw text pieces during streaming.
        """

        # Run table search to get sources and format the prompt
        sources = self.search(user_message, table=self.table, limit=self.search_limit)
        prompt = self.generation_engine.format_prompt(user_message, sources)

        # Generate token by-token using the generation engine
        for piece in self.generation_engine._generate_llama_cpp_stream(prompt):
            if logger.isEnabledFor(logging.DEBUG):
                print(piece, end="", flush=True)  # Print the raw text piece
            yield piece

        # After the stream finishes, return the final processed sections
        raw = self.generation_engine._last_raw
        sections = self.generation_engine.extract_sections(raw)
        if "answer" in sections:
            info = self.generation_engine.extract_citations(sections["answer"], sources)
            sections["clean_answer"] = info["clean_text"]
            sections["citations"] = info["citations"]
        yield {"__done__": sections, "sources": sources}
