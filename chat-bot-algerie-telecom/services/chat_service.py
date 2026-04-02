"""
ChatService — Advanced RAG Orchestration with Citation Enforcement
===================================================================
Full pipeline:

1. **Semantic cache** check (sub-ms for repeated / paraphrase queries)
2. **Over-fetch** top-20 from Qdrant vector store
3. **Cross-encoder rerank** → top-10 precision passages
4. **MMR selection** → top-5 diverse passages (no redundancy)
5. **Context compression** (sentence-level extraction, ~60 % token savings)
6. **Lost-in-the-middle reorder** (best evidence at start + end)
7. **LLM generation** with strict citation-faithful prompt
8. **Cache storage** for future queries
9. **Rich source attribution** with filenames, pages, scores, S3 keys

Prompt Strategy
---------------
Every response MUST cite the internal Algérie Télécom documents retrieved
during context gathering. The prompt template enforces:
- ``[Source: <document>, Page <N>, Article <X>]`` after each factual claim
- No information is generated that cannot be traced to a retrieved document
- Sources are attached to the response for frontend display
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

from components.llm.llm_component import LLMComponent
from components.embedding.embedding_component import EmbeddingComponent
from components.vector_store.vector_store_component import (
    VectorStoreComponent,
    VectorDocument,
)

logger = logging.getLogger("forsa.services.chat")


# ── RAG enhancements (graceful fallback) ────────────────────────────────────
try:
    from rag_enhancements import (
        SemanticCache,
        ContextCompressor,
        MMRSelector,
        LostInMiddleReorderer,
        SharedCrossEncoderReranker,
    )

    _ENHANCEMENTS = True
except ImportError:
    _ENHANCEMENTS = False
    logger.warning("rag_enhancements not available — running basic RAG pipeline")


# ---------------------------------------------------------------------------
# Citation-Faithful System Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_TEMPLATE = """Tu es un assistant juridique et administratif spécialisé dans les documents internes d'Algérie Télécom.

RÈGLE ABSOLUE DE CITATION :
Chaque affirmation factuelle DOIT être suivie immédiatement d'une citation au format exact :
  [Source: <nom_du_document>, Page <N>, Article <X>]

Si tu n'as pas de source précise pour une information, n'inclus PAS cette information.
N'invente AUCUNE citation. Utilise UNIQUEMENT les informations des documents fournis ci-dessous.

FORMAT DE RÉPONSE OBLIGATOIRE :
1. Commence directement par la réponse (sans introduction).
2. Chaque fait est suivi de sa citation entre crochets.
3. Exemple :
   La pénalité de retard est de **0,5% par jour** [Source: convention_P.docx, Page 4, Article 12].
   Les bénéficiaires sont les cadres supérieurs et retraités [Source: convention_P.docx, Page 1, Article 2].

Si aucun document pertinent n'est trouvé :
  "Aucune information correspondante dans les documents fournis."

Ne reformule pas les articles - cite-les précisément.
Réponds en {language}."""

_GENERAL_SYSTEM_PROMPT = """Tu es un assistant intelligent d'Algérie Télécom. Tu aides les utilisateurs avec leurs questions sur les offres, les guides, les conventions et les produits d'Algérie Télécom.

Réponds de manière précise et professionnelle. Si tu disposes de documents de référence, cite-les.
Si tu ne connais pas la réponse, dis-le honnêtement.

Réponds en {language}."""


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
_OVERFETCH_K = 20        # Retrieve more from Qdrant for reranking headroom
_RERANK_TOP_K = 10       # Keep top-10 after cross-encoder
_MMR_TOP_K = 5           # Keep top-5 diverse passages for LLM context
_COMPRESS_SENTENCES = 8  # Max sentences per passage after compression
_CACHE_SIZE = 512        # Semantic cache entries
_CACHE_TTL = 1800.0      # Cache TTL in seconds (30 min)


class ChatService:
    """
    Advanced RAG orchestration service.

    Injected dependencies:
    - LLMComponent for generation
    - EmbeddingComponent for query embedding
    - VectorStoreComponent for document retrieval
    """

    def __init__(
        self,
        llm: LLMComponent,
        embedding: EmbeddingComponent,
        vector_store: VectorStoreComponent,
    ) -> None:
        self._llm = llm
        self._embedding = embedding
        self._vector_store = vector_store

        # Initialise RAG enhancement components (singletons where possible)
        if _ENHANCEMENTS:
            self._cache = SemanticCache(
                max_size=_CACHE_SIZE, ttl_seconds=_CACHE_TTL
            )
            self._compressor = ContextCompressor(
                max_sentences=_COMPRESS_SENTENCES
            )
            self._mmr = MMRSelector(lambda_factor=0.7)
            self._reranker = SharedCrossEncoderReranker()
            self._reorderer = LostInMiddleReorderer()
        else:
            self._cache = None
            self._compressor = None
            self._mmr = None
            self._reranker = None
            self._reorderer = None

    # ------------------------------------------------------------------
    # OpenAI-Compatible Chat Completion
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        use_context: bool = True,
        context_filter: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        language: str = "français",
    ) -> Dict[str, Any]:
        """
        Process a chat completion request with the full advanced RAG pipeline.

        Pipeline stages (when enhancements are available):
        1. Cache check
        2. Over-fetch top-20 from Qdrant
        3. Cross-encoder rerank → top-10
        4. MMR diversity selection → top-5
        5. Context compression (sentence extraction)
        6. Lost-in-the-middle reordering
        7. LLM generation with citation prompt
        8. Cache storage + return with rich sources
        """
        t0 = time.perf_counter()

        # Extract the last user message for retrieval
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return self._build_response(
                content="No user message provided.",
                sources=[],
                latency_ms=0,
            )

        # ── Stage 0: Semantic Cache ──────────────────────────────────
        cache_key = f"chat::{user_message}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.info("Cache HIT (%.1f ms): %.60s", latency_ms, user_message)
                cached["latency_ms"] = round(latency_ms, 1)
                return cached

        # ── Stage 1: Retrieve from vector store (over-fetch) ────────
        sources = []
        augmented_messages = list(messages)

        if use_context and self._vector_store.count() > 0:
            fetch_k = _OVERFETCH_K if self._reranker else 5
            retrieved_docs = self._vector_store.search(
                query=user_message,
                top_k=fetch_k,
                filter_metadata=context_filter,
            )

            if retrieved_docs:
                # Convert VectorDocuments to passage dicts for processing
                passages = self._docs_to_passages(retrieved_docs)

                # ── Stage 2: Cross-encoder reranking ─────────────────
                if self._reranker is not None and len(passages) > _MMR_TOP_K:
                    passages = self._reranker.rerank(
                        query=user_message,
                        passages=passages,
                        text_field="text",
                        top_k=_RERANK_TOP_K,
                    )
                    logger.debug(
                        "Cross-encoder reranked %d → %d passages",
                        len(retrieved_docs), len(passages),
                    )

                # ── Stage 3: MMR diversity selection ─────────────────
                if self._mmr is not None and len(passages) > _MMR_TOP_K:
                    relevance_scores = [
                        p.get("_ce_score", p.get("score", 0.0))
                        for p in passages
                    ]
                    passages = self._mmr.select(
                        passages=passages,
                        query=user_message,
                        top_k=_MMR_TOP_K,
                        text_field="text",
                        relevance_scores=relevance_scores,
                    )
                    logger.debug("MMR selected %d diverse passages", len(passages))
                else:
                    passages = passages[:_MMR_TOP_K]

                # ── Stage 4: Context compression ─────────────────────
                if self._compressor is not None:
                    passages = self._compressor.compress(
                        passages, user_message, text_field="text"
                    )
                    logger.debug("Compressed passages to key sentences")

                # ── Stage 5: Lost-in-the-middle reordering ───────────
                if self._reorderer is not None:
                    passages = self._reorderer.reorder(passages)

                # Build the context block and source list
                context_block = self._build_context_block(
                    passages, user_message, language
                )
                system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(language=language)
                augmented_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_block},
                ]

                if include_sources:
                    sources = self._build_sources(passages)
        else:
            # No vector store data — use general prompt
            if not any(m.get("role") == "system" for m in augmented_messages):
                system_prompt = _GENERAL_SYSTEM_PROMPT.format(language=language)
                augmented_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

        # ── Stage 6: Generate response ───────────────────────────────
        if stream:
            return self._stream_response(
                augmented_messages, sources, max_new_tokens, temperature, top_p
            )

        response_text = self._llm.generate_chat(
            augmented_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        result = self._build_response(
            content=response_text,
            sources=sources,
            latency_ms=latency_ms,
        )

        # ── Stage 7: Cache storage ───────────────────────────────────
        if self._cache is not None:
            self._cache.put(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Legacy Pipeline Bridge
    # ------------------------------------------------------------------

    def process_with_pipeline(
        self,
        query: str,
        category: str,
        pipeline_fn,
    ) -> Dict[str, Any]:
        """
        Bridge method for the legacy ``/process-question`` endpoint.

        Delegates to the existing pipeline functions while they are being
        migrated to the new architecture.
        """
        return pipeline_fn(query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docs_to_passages(docs: List[VectorDocument]) -> List[Dict[str, Any]]:
        """Convert VectorDocument objects into uniform passage dicts."""
        passages = []
        for doc in docs:
            passage = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "score": doc.score,
                # Carry all metadata through the pipeline
                "filename": (
                    doc.metadata.get("filename")
                    or doc.metadata.get("doc_name")
                    or doc.doc_id
                ),
                "page": doc.metadata.get("page", "?"),
                "category": doc.metadata.get("category", ""),
                "s3_key": doc.metadata.get("s3_key", ""),
                "article": doc.metadata.get("article_ref", ""),
            }
            # Preserve any extra metadata
            for k, v in doc.metadata.items():
                if k not in passage:
                    passage[k] = v
            passages.append(passage)
        return passages

    @staticmethod
    def _build_sources(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build rich source attribution list from processed passages.

        Each source includes enough metadata for the frontend to:
        - Display the document name and page
        - Link to the original file (S3 key)
        - Show the relevance score
        """
        sources = []
        seen = set()

        for i, p in enumerate(passages):
            # Determine the best available score
            score = p.get("_ce_score", p.get("score", 0.0))
            filename = p.get("filename", p.get("doc_id", "unknown"))
            page = p.get("page", "?")
            source_key = f"{filename}::{page}"

            # Deduplicate same file + page
            if source_key in seen:
                continue
            seen.add(source_key)

            text_preview = (p.get("text") or "")[:250]
            if len(p.get("text", "")) > 250:
                text_preview += "…"

            sources.append({
                "rank": i + 1,
                "doc_id": p.get("doc_id", ""),
                "filename": filename,
                "page": page,
                "category": p.get("category", ""),
                "article": p.get("article", ""),
                "s3_key": p.get("s3_key", ""),
                "relevance_score": round(float(score), 4),
                "text_preview": text_preview,
            })

        return sources

    def _build_context_block(
        self,
        passages: List[Dict[str, Any]],
        query: str,
        language: str,
    ) -> str:
        """Build the context block injected into the prompt."""
        lines = [f"=== QUESTION ===\n{query}\n"]
        lines.append(
            "=== DOCUMENTS RÉCUPÉRÉS (utilise UNIQUEMENT ces sources) ===\n"
        )

        for i, p in enumerate(passages, 1):
            filename = p.get("filename", p.get("doc_id", "unknown"))
            page = p.get("page", "?")
            category = p.get("category", "")
            article = p.get("article", "")
            score = p.get("_ce_score", p.get("score", 0.0))

            lines.append(f"--- Document {i}: {filename} ---")
            lines.append(f"    Catégorie: {category}")
            lines.append(f"    Page: {page}")
            if article:
                lines.append(f"    Article: {article}")
            lines.append(f"    Score de pertinence: {score:.4f}")
            lines.append(f"    Contenu:")

            # Truncate content to keep within token budget
            text = p.get("text", "")
            if len(text) > 2000:
                text = text[:2000] + "… [tronqué]"
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def _build_response(
        self,
        content: str,
        sources: List[Dict[str, Any]],
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible response dict with source attribution."""
        return {
            "id": f"chatcmpl-forsa-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._llm.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Not tracked at this level
                "completion_tokens": -1,
                "total_tokens": -1,
            },
            # Forsa-specific extensions — rich source attribution
            "sources": sources,
            "latency_ms": round(latency_ms, 1),
        }

    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        sources: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> Generator[str, None, None]:
        """Generate SSE-compatible streaming chunks."""
        import json as _json

        stream_id = f"chatcmpl-forsa-{int(time.time() * 1000)}"

        # Send sources as the first SSE event so the frontend can display them
        # while the LLM is still generating
        if sources:
            source_event = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self._llm.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
                "sources": sources,
            }
            yield f"data: {_json.dumps(source_event)}\n\n"

        token_stream = self._llm.generate_chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

        for chunk_text in token_stream:
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self._llm.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {_json.dumps(chunk)}\n\n"

        # Final chunk
        final = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self._llm.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {_json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
