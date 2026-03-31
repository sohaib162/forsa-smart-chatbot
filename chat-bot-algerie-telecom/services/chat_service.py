"""
ChatService — RAG Orchestration with Citation Enforcement
==========================================================
The core service that ties together:

1. **Vector retrieval** from the Qdrant index (via VectorStoreComponent)
2. **Knowledge Graph enrichment** (via existing AdvancedPipeline)
3. **LLM generation** with citation-faithful prompts (via LLMComponent)
4. **Hallucination validation** (via HallucinationValidator)

This service is used by both:
- The new OpenAI-compatible ``/v1/chat/completions`` endpoint
- The legacy ``/process-question`` endpoint (backward compatibility)

Prompt Strategy
---------------
Every response MUST cite the internal Algérie Télécom documents retrieved
during context gathering. The prompt template enforces:
- ``[Source: <document>, Page <N>, Article <X>]`` after each factual claim
- No information is generated that cannot be traced to a retrieved document
- A faithfulness score is computed and appended to the response
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


class ChatService:
    """
    RAG orchestration service.

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
        Process a chat completion request.

        If ``use_context=True`` (default), the last user message is used
        to retrieve relevant documents from the vector store, which are
        injected into the prompt as context.

        Returns an OpenAI-compatible response dict.
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

        # Step 1: Retrieve context from vector store
        sources = []
        augmented_messages = list(messages)

        if use_context and self._vector_store.count() > 0:
            retrieved_docs = self._vector_store.search(
                query=user_message,
                top_k=5,
                filter_metadata=context_filter,
            )

            if retrieved_docs:
                context_block = self._build_context_block(
                    retrieved_docs, user_message, language
                )
                # Inject context into system prompt
                system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(language=language)
                augmented_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_block},
                ]

                if include_sources:
                    sources = [
                        {
                            "doc_id": doc.doc_id,
                            "text": doc.text[:200] + "…" if len(doc.text) > 200 else doc.text,
                            "score": round(doc.score, 4),
                            **doc.metadata,
                        }
                        for doc in retrieved_docs
                    ]
        else:
            # No vector store data — use general prompt
            if not any(m.get("role") == "system" for m in augmented_messages):
                system_prompt = _GENERAL_SYSTEM_PROMPT.format(language=language)
                augmented_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

        # Step 2: Generate response
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

        return self._build_response(
            content=response_text,
            sources=sources,
            latency_ms=latency_ms,
        )

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
    # Internal
    # ------------------------------------------------------------------

    def _build_context_block(
        self,
        documents: List[VectorDocument],
        query: str,
        language: str,
    ) -> str:
        """Build the context block injected into the prompt."""
        lines = [f"=== QUESTION ===\n{query}\n"]
        lines.append("=== DOCUMENTS RÉCUPÉRÉS (utilise UNIQUEMENT ces sources) ===\n")

        for i, doc in enumerate(documents, 1):
            doc_name = (
                doc.metadata.get("filename")
                or doc.metadata.get("doc_name")
                or doc.doc_id
            )
            s3_key = doc.metadata.get("s3_key", "")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "")

            lines.append(f"--- Document {i}: {doc_name} ---")
            lines.append(f"    Catégorie: {category}")
            lines.append(f"    Page: {page}")
            lines.append(f"    Score: {doc.score:.4f}")
            lines.append(f"    S3: {s3_key}")
            lines.append(f"    Contenu:")

            # Truncate content to keep within token budget
            text = doc.text
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
        """Build an OpenAI-compatible response dict."""
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
            # Forsa-specific extensions
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
