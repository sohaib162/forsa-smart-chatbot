"""
Local Qwen 2.5 3B LLM Client — DI-Aware Singleton
====================================================
Instead of loading a second copy of the model (which causes CUDA OOM on
GPUs with ≤6 GB VRAM), this module now delegates to the LLMComponent
already loaded by the DI container.

The public API is unchanged:
  - get_llm_client()           → returns an object with .generate() and .generate_with_citations()
  - call_local_llm(...)        → backward-compatible convenience function
  - call_local_llm_with_citations(...)
  - preload_llm()              → no-op (model is loaded by DI container)

All existing pipelines continue to work without modification.
"""
import logging
from typing import Optional

logger = logging.getLogger("forsa.llm")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_llm_client = None


def get_llm_client():
    """Get the LLM client singleton.

    Reuses the LLMComponent already loaded by the DI container to avoid
    loading the model twice (which causes CUDA OOM on small GPUs).
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        from di import global_injector
        from components.llm.llm_component import LLMComponent

        if global_injector.is_bootstrapped and global_injector.has(LLMComponent):
            _llm_client = global_injector.get(LLMComponent)
            logger.info("LLM client: reusing DI-managed LLMComponent (no extra VRAM)")
            return _llm_client
    except Exception as e:
        logger.warning("Could not retrieve LLMComponent from DI: %s", e)

    # Fallback: if DI is not bootstrapped (e.g. running a standalone script),
    # load locally.  This path should NOT be hit during normal server operation.
    logger.warning(
        "DI container not ready — falling back to standalone model load. "
        "This will use additional GPU memory!"
    )
    from _local_llm_standalone import LocalLLMClient
    _llm_client = LocalLLMClient()
    return _llm_client


def preload_llm():
    """Pre-load the LLM model.

    With the DI architecture this is effectively a no-op because the model
    is already loaded by the DI container's bootstrap sequence.  We still
    call get_llm_client() to populate the module-level cache.
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
    return client.generate(system_prompt, user_content, max_new_tokens=max_new_tokens)


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
