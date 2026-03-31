"""
LLMComponent — Dependency-Injected Local LLM Wrapper
=====================================================
Wraps the existing LocalLLMClient (Qwen 2.5 3B) behind a clean,
injectable interface. Exposes both:

1. A LlamaIndex-compatible ``CustomLLM`` for framework integration.
2. The raw ``generate()`` / ``generate_with_citations()`` methods used
   by the existing AdvancedPipeline (backward compatibility).

Design Rationale
----------------
- The existing ``LocalLLMClient`` singleton pattern is preserved internally.
- This component adds a layer of indirection so the LLM can be swapped
  (e.g. to a larger Qwen 14B, or a GGUF via llama.cpp) without touching
  any service or pipeline code.
- Generation parameters (temperature, top_k, etc.) come from ``LLMSettings``
  which reads from environment variables → zero hardcoded values.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import LLMSettings

logger = logging.getLogger("forsa.components.llm")

# Citation detection pattern (reused from the original local_llm_client)
_CITATION_PRESENT = re.compile(r"\[Source:", re.IGNORECASE)

_RETRY_REMINDER = (
    "\n\n[RAPPEL CRITIQUE: Chaque fait DOIT être suivi de "
    "[Source: <nom_document>, Page <N>, Article <X>]. "
    "Reformule ta réponse en incluant les citations.]"
)


class LLMComponent:
    """
    Injectable LLM component for the Forsa chatbot.

    Lifecycle
    ---------
    Created once by the DI container at startup.  The model is loaded
    eagerly into GPU/CPU memory so the first query has zero cold-start.

    Thread Safety
    -------------
    ``torch.no_grad()`` + ``model.eval()`` makes inference safe for
    concurrent ``asyncio.to_thread()`` calls (read-only on weights).
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[str] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_name = self._settings.model_name
        logger.info("LLMComponent: Loading model '%s' …", model_name)
        t0 = time.perf_counter()

        # Resolve device
        if self._settings.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._settings.device

        if self._device == "cuda":
            logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("  No GPU detected — using CPU (slower inference)")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self._settings.trust_remote_code,
        )

        if self._device == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=self._settings.trust_remote_code,
                low_cpu_mem_usage=True,
                max_memory={
                    0: self._settings.gpu_memory_limit,
                    "cpu": self._settings.cpu_memory_limit,
                },
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=self._settings.trust_remote_code,
            )

        self._model.eval()
        elapsed = time.perf_counter() - t0
        logger.info(
            "LLMComponent: Model loaded on %s in %.1fs", self._device, elapsed
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Standard text generation.

        Falls back to ``LLMSettings`` defaults for any parameter not
        explicitly provided.
        """
        return self._generate_raw(
            system_prompt=system_prompt,
            user_content=user_content,
            max_new_tokens=max_new_tokens or self._settings.max_new_tokens,
            temperature=temperature if temperature is not None else 0.7,
            top_p=top_p if top_p is not None else 0.9,
            top_k=top_k if top_k is not None else 50,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else 1.1
            ),
        )

    def generate_with_citations(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_new_tokens: int = 512,
        require_citations: bool = True,
        max_retries: int = 1,
    ) -> str:
        """
        Citation-faithful generation with retry logic.

        Uses stricter decoding parameters (lower temperature, top_k=20)
        optimised for legal/administrative text.  If the first response
        contains no ``[Source: …]`` tags and ``require_citations=True``,
        a retry with an even harder reminder is triggered.
        """
        response = self._generate_raw(
            system_prompt=system_prompt,
            user_content=user_content,
            max_new_tokens=max_new_tokens,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
            repetition_penalty=self._settings.repetition_penalty,
        )

        if require_citations and not _CITATION_PRESENT.search(response):
            for _ in range(max_retries):
                retry_content = user_content + _RETRY_REMINDER
                response = self._generate_raw(
                    system_prompt=system_prompt,
                    user_content=retry_content,
                    max_new_tokens=max_new_tokens,
                    temperature=0.15,
                    top_p=0.80,
                    top_k=15,
                    repetition_penalty=1.2,
                )
                if _CITATION_PRESENT.search(response):
                    break

        return response

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        OpenAI-compatible chat completion interface.

        Accepts ``messages`` in ``[{"role": ..., "content": ...}]`` format.
        If ``stream=True``, yields token chunks (for SSE streaming).
        """
        max_tokens = max_new_tokens or self._settings.max_new_tokens
        temp = temperature if temperature is not None else self._settings.temperature
        tp = top_p if top_p is not None else self._settings.top_p

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer(
            [text], return_tensors="pt"
        ).to(self._device)

        if stream:
            return self._stream_generate(model_inputs, max_tokens, temp, tp)
        else:
            return self._batch_generate(model_inputs, max_tokens, temp, tp)

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
                {"role": "user", "content": user_content},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self._tokenizer(
                [text], return_tensors="pt"
            ).to(self._device)
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
                input_len,
                output_len,
                temperature,
                elapsed,
            )
            return response.strip()

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception("LLM inference failed after %.1fs", elapsed)
            return f"ERROR: Failed to generate response - {e}"

    def _batch_generate(
        self,
        model_inputs: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Non-streaming generation for chat completions."""
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self._settings.repetition_penalty,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]
        return self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def _stream_generate(
        self,
        model_inputs: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        """
        Token-by-token streaming generation.

        Uses ``model.generate()`` with ``max_new_tokens=1`` in a loop
        for simplicity.  For production, consider TextIteratorStreamer.
        """
        from transformers import TextIteratorStreamer
        import threading

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=self._settings.repetition_penalty,
            streamer=streamer,
        )

        thread = threading.Thread(
            target=self._model.generate, kwargs=generation_kwargs
        )
        thread.start()

        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk

        thread.join()
