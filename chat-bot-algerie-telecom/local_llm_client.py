"""
Local Qwen 2.5 3B LLM Client
Uses HuggingFace Transformers for local inference.

Extended with:
  - generate_with_citations(): stricter decoding settings for legal/admin text
  - retry logic: if the first response has zero [Source: …] citations and
    citations are expected, the model is called once more with an even more
    direct reminder.
"""
import os
import logging
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

logger = logging.getLogger("forsa.llm")

# Pattern to detect at least one citation block in the response
_CITATION_PRESENT = re.compile(r"\[Source:", re.IGNORECASE)

# Retry prompt appended when no citation is found on the first attempt
_RETRY_REMINDER = (
    "\n\n[RAPPEL CRITIQUE: Chaque fait DOIT être suivi de "
    "[Source: <nom_document>, Page <N>, Article <X>]. "
    "Reformule ta réponse en incluant les citations.]"
)


class LocalLLMClient:
    """Singleton client for local Qwen 2.5 3B model."""

    _instance: Optional["LocalLLMClient"] = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._initialize_model()

    def _initialize_model(self):
        model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

        logger.info("Loading model: %s", model_name)
        t0 = time.perf_counter()

        if torch.cuda.is_available():
            self._device = "cuda"
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            self._device = "cpu"
            logger.warning("No GPU detected — using CPU (inference will be slower)")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        if self._device == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "4.5GB", "cpu": "8GB"},
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        self._model.eval()
        load_time = time.perf_counter() - t0
        logger.info("Model loaded on %s in %.1fs", self._device, load_time)

    # ------------------------------------------------------------------
    # Original generation method (unchanged for backward compatibility)
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Standard generation.  Same interface as before.
        Used by legacy pipelines that have not yet been upgraded.
        """
        return self._generate_raw(
            system_prompt=system_prompt,
            user_content=user_content,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # ------------------------------------------------------------------
    # Citation-faithful generation (used by AdvancedPipeline)
    # ------------------------------------------------------------------

    def generate_with_citations(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int = 512,
        require_citations: bool = True,
        max_retries: int = 1,
    ) -> str:
        """
        Generation tuned for citation-faithful legal/admin responses.

        Key differences from ``generate()``:
        - Lower temperature (0.2) → more deterministic, less hallucination
        - Higher repetition_penalty (1.15) → avoids repeating the same phrase
        - top_k=20 → limits vocabulary to high-probability tokens
        - If ``require_citations=True`` and the first response contains no
          [Source: …] tags, a single retry is triggered with a hard reminder
          appended to the user content.

        Parameters
        ----------
        system_prompt     : the citation-faithful system prompt
        user_content      : context block built by _build_citation_context_block
        max_new_tokens    : max tokens (512 recommended for verbose conventions)
        require_citations : whether to retry if no citations found
        max_retries       : max number of retry attempts (default 1)

        Returns
        -------
        The best response string (may include [Source: …] citations).
        """
        response = self._generate_raw(
            system_prompt=system_prompt,
            user_content=user_content,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.85,
            top_k=20,
            repetition_penalty=1.15,
        )

        # Retry once if citations are required but none found
        if require_citations and not _CITATION_PRESENT.search(response):
            for _ in range(max_retries):
                retry_content = user_content + _RETRY_REMINDER
                response = self._generate_raw(
                    system_prompt=system_prompt,
                    user_content=retry_content,
                    max_new_tokens=max_new_tokens,
                    temperature=0.15,   # even lower on retry
                    top_p=0.80,
                    top_k=15,
                    repetition_penalty=1.2,
                )
                if _CITATION_PRESENT.search(response):
                    break

        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_raw(
        self,
        system_prompt: str,
        user_content: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        t0 = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self._tokenizer([text], return_tensors="pt").to(
                self._device
            )
            input_len = model_inputs.input_ids.shape[-1]

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids
                )
            ]
            output_len = len(generated_ids[0])
            response = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            elapsed = time.perf_counter() - t0
            logger.info(
                "LLM inference | in=%d tok | out=%d tok | temp=%.2f | %.1fs",
                input_len, output_len, temperature, elapsed,
            )
            return response.strip()

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("LLM inference failed after %.1fs", elapsed)
            return f"ERROR: Failed to generate response - {str(e)}"


# ---------------------------------------------------------------------------
# Module-level singletons and helpers
# ---------------------------------------------------------------------------

_llm_client: Optional[LocalLLMClient] = None


def get_llm_client() -> LocalLLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


def preload_llm() -> LocalLLMClient:
    """Eagerly load the LLM model at application startup.

    Call this from main.py at import time or in lifespan so the model
    is already on the GPU when the first query arrives.
    """
    logger.info("Pre-loading LLM model for instant first-query response...")
    client = get_llm_client()
    logger.info("LLM pre-loaded and ready.")
    return client


def call_local_llm(
    system_prompt: str,
    user_content: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Standard LLM call (backward-compatible).
    Used by existing pipelines that have not been upgraded to AdvancedPipeline.
    """
    client = get_llm_client()
    return client.generate(system_prompt, user_content, max_new_tokens)


def call_local_llm_with_citations(
    system_prompt: str,
    user_content: str,
    max_new_tokens: int = 512,
    require_citations: bool = True,
) -> str:
    """
    Citation-faithful LLM call.
    Called by AdvancedPipeline; can also be called directly by upgraded pipelines.
    """
    client = get_llm_client()
    return client.generate_with_citations(
        system_prompt=system_prompt,
        user_content=user_content,
        max_new_tokens=max_new_tokens,
        require_citations=require_citations,
    )
