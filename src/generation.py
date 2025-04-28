import logging
import re
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class GenerationEngine:
    def __init__(
        self,
        model_path_or_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        trust_remote_code: bool = True,
        backend: str = "llama_cpp",
    ):
        """
        Initialize the RAG Generator with llama.cpp.

        Args:
            model_path_or_name: Path to the model, HuggingFace model name, or name from available models:
                               - "1b_rag": PleIAs/1b_rag_traceback
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter (lower = more focused)
            repetition_penalty: Repetition penalty to avoid loops
            trust_remote_code: Whether to trust remote code in model repo
            hf_token: Hugging Face API token (required if using predefined model names)
            models_dir: Directory where models will be stored (default: ./pleias_models)
            backend: Optional backend to use for model loading. Here, only llama.cpp is available.
        """
        # Check if this is a predefined model name

        self.model_path = model_path_or_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.trust_remote_code = trust_remote_code

        self.backend = backend
        self._init_llama_cpp()

    #################################
    # Model Initialization Methods  #
    #################################

    def _init_llama_cpp(self):
        """
        Initialize the model using llama_cpp
        """
        from llama_cpp import Llama

        logger.info("Loading model with llama_cpp")

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False,
        )
        logger.info("Model loaded successfully")

    ###################################
    # Prompt Formatting and Generation #
    ###################################

    def format_prompt(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """
        Format the query and sources into a prompt with special tokens.

        The prompt follows a specific format with special tokens to guide the model:
        - <|query_start|>...<|query_end|> for the user's question
        - <|source_start|><|source_id|>N ...<|source_end|> for each source
        - <|language_start|> to indicate the beginning of generation
        Args:
            query: The user's question
            sources: List of source documents with their metadata. Format is list of dictionaries,
                     each with a "text" key and optional "metadata" key.
                     The metadata is not used in the prompt but can be useful for later processing.
                     Example: [{"text": "Document text", "metadata": {"source_id": 1, "source_name": "Doc1"}}]
        Returns:
            Formatted prompt string
        """
        prompt = f"<|query_start|>{query}<|query_end|>\n"

        # Add each source with its ID
        for idx, source in enumerate(sources, 1):
            source_text = source.get("text", "")
            prompt += (
                f"<|source_start|><|source_id|>{idx} {source_text}<|source_end|>\n"
            )

        # Add the source analysis start token
        prompt += "<|language_start|>\n"

        logger.debug(f"Formatted prompt: \n {prompt}")
        return prompt

    def _generate_llama_cpp(self, formatted_prompt: str) -> str:
        """
        Generate text using llama_cpp backend.
        This method handles text generation when using the llama_cpp backend (CPU)

        Args:
            formatted_prompt: The properly formatted input prompt

        Returns:
            Generated text response
        """
        t0 = time.time()

        tokens = self.model.generate(
            self.model.tokenize(formatted_prompt.encode("utf-8"), special=True),
            temp=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            reset=True,
        )
        generated_text = ""
        t1 = None

        for i, t in enumerate(tokens):
            # Compute prefill time
            if t1 is None:
                t1 = time.time()
                logger.info(f"Prefill time: {t1 - t0:.2f} seconds")

            piece = self.model.detokenize([t], special=True).decode(
                "utf-8", errors="replace"
            )
            if (piece == "<|end_of_text|>") | (i >= self.max_tokens):
                break
            generated_text += piece

        t2 = time.time()
        logger.info(f"Generation time: {t2 - t1:.2f} seconds")

        return generated_text.strip()

    def _generate_llama_cpp_stream(self, formatted_prompt: str) -> str:
        """
        Generates a stream of text using the LLaMA model with the given formatted prompt.
        This function tokenizes the input prompt, generates tokens using the model, and detokenizes
        them to produce a stream of text. It logs the prefill time (time taken to generate the first token)
        and stops generating when the end-of-text token is encountered or the maximum token limit is reached.
        Args:
            formatted_prompt (str): The input prompt formatted as a string.
        Yields:
            str: A piece of the generated text.
        """

        t0 = time.time()
        tokens = self.model.generate(
            self.model.tokenize(formatted_prompt.encode("utf-8"), special=True),
            temp=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            reset=True,
        )
        t1 = None
        self._last_raw = ""

        for i, t in enumerate(tokens):
            # Compute prefill time
            if t1 is None:
                t1 = time.time()
                logger.info(f"Prefill time: {t1 - t0:.2f} seconds")

            piece = self.model.detokenize([t], special=True).decode(
                "utf-8", errors="replace"
            )
            self._last_raw += piece
            if (piece == "<|end_of_text|>") | (i >= self.max_tokens):
                break
            yield piece

    #############################
    # Response Processing       #
    #############################

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract all sections from the generated text using the output format.

        The model's output is structured with section markers that need to be parsed.
        This method extracts different sections like query_analysis, query_report,
        source_analysis, draft, and answer.

        Note: query_analysis is included in the prompt, not in the output.
        Args:
            text: The generated text response
        Returns:
            Dictionary with all extracted sections
        """
        result = {}

        # For language, we need to handle it differently since it's in the prompt
        # Extract everything from the start until query_analysis_end
        language_end_match = re.search(r"<\|language_end\|>", text, re.DOTALL)
        if language_end_match:
            end_pos = language_end_match.start()
            result["language"] = text[:end_pos].strip()

        # Define other section patterns to extract
        section_patterns = {
            "query_report": r"<\|query_report_start\|>(.*?)<\|query_report_end\|>",
            "source_analysis": r"<\|source_analysis_start\|>(.*?)<\|source_analysis_end\|>",
            "draft": r"<\|draft_start\|>(.*?)<\|draft_end\|>",
            "answer": r"<\|answer_start\|>(.*?)<\|answer_end\|>",
        }

        # Extract each section using regex
        for section_name, pattern in section_patterns.items():
            section_match = re.search(pattern, text, re.DOTALL)
            if section_match:
                result[section_name] = section_match.group(1).strip()

        # If no sections were found, return the full text
        if not result:
            result["full_text"] = text

        logger.info("Extracted sections")
        return result

    def extract_citations(
        self, answer: str, sources: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract citations from the answer and format them with numbered references.

        Args:
            answer: The answer text containing citations
            sources: List of source documents (optional)

        Returns:
            Dictionary with clean text and citations data
        """
        # Pattern to match <ref name="<|source_id|>NUMBER">text</ref>
        citation_pattern = r'<ref name="(?:<\|source_id\|>)?(\d+)">(.*?)<\/ref>'

        # Create a working copy and citation list
        clean_text = answer
        citations = []

        # Find all citations and process them one by one
        citation_count = 0
        while True:
            match = re.search(citation_pattern, clean_text)
            if not match:
                break

            citation_count += 1
            source_id = match.group(1)
            cited_text = match.group(2)
            full_match = match.group(0)
            start_pos = match.start()

            # Get some context for supported_text (look back up to 150 chars for context)
            text_before = clean_text[:start_pos]
            sentence_boundary = max(
                text_before.rfind(". "),
                text_before.rfind("! "),
                text_before.rfind("? "),
                text_before.rfind("\n"),
            )

            if sentence_boundary == -1:
                supported_text = text_before[-min(150, len(text_before)) :].strip()
            else:
                supported_text = text_before[sentence_boundary + 2 :].strip()

            # Replace this citation with a numbered reference
            clean_text = clean_text.replace(full_match, f"[{citation_count}]", 1)

            # Store citation data
            citations.append(
                {
                    "citation_number": citation_count,
                    "source_id": source_id,
                    "cited_text": cited_text,
                    "supported_text": supported_text,
                }
            )

        # If no citations found, return the original text
        if not citations:
            return {"clean_text": answer, "citations": []}

        # Create citations section
        citations_section = "\n\n**Citations**\n"
        for citation in citations:
            citations_section += f'[{citation["citation_number"]}] "{citation["cited_text"]}" [Source {citation["source_id"]}]\n'

        # Add citations section to clean text
        final_text = clean_text + citations_section

        logger.info("Citations extracted")
        return {"clean_text": final_text, "citations": citations}

    #############################
    # Main Interface Methods    #
    #############################

    def generate(self, query: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response based on the query and sources.

        This is the main method to use for generating responses. It:
        1. Formats the prompt
        2. Generates text using the appropriate backend
        3. Processes the response to extract sections
        4. Extracts and formats citations
        Args:
            query: The user's question
            sources: List of source documents with their metadata
        Returns:
            Dictionary with raw response and processed sections
        """
        formatted_prompt = self.format_prompt(query, sources)

        # Generate response using the appropriate backend
        raw_response = self._generate_llama_cpp(formatted_prompt)

        # Process the response
        sections = self.extract_sections(raw_response)

        # Extract citations if answer section exists
        if "answer" in sections:
            citation_info = self.extract_citations(sections["answer"], sources)
            sections["clean_answer"] = citation_info["clean_text"]
            sections["citations"] = citation_info["citations"]

        response = {
            "raw_response": raw_response,
            "processed": sections,
            "sources": sources,
            "backend_used": self.backend,
        }
        return response
