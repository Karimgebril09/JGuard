from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

RELEVANCE_THRESHOLD = 0.15
REDUNDANCY_THRESHOLD = 0.85


class WebSearchGuard:
    def __init__(self) -> None:
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _cosine_sim(self, text_a: str, text_b: str) -> float:
        embedder = self._get_embedder()
        vecs = embedder.encode([text_a, text_b], convert_to_numpy=True)
        return float(sk_cosine_similarity(vecs[0:1], vecs[1:2])[0][0])

    def check_query(
        self,
        query: str,
        original_request: str,
        past_queries: list[str],
        tool_call_id: str,
    ) -> tuple[bool, dict[str, Any]]:
        if not query:
            return False, {}

        relevance = self._cosine_sim(query, original_request)
        print(f"[WebSearchGuard] Query: '{query}' | Relevance to original: {relevance:.3f}")
        if relevance < RELEVANCE_THRESHOLD:
            print(
                f"[WebSearchGuard] BLOCKED: off-topic "
                f"(sim={relevance:.3f} < threshold={RELEVANCE_THRESHOLD})"
            )
            return True, {
                "messages": [
                    ToolMessage(
                        content=(
                            f"[SEARCH BLOCKED: Off-topic] The query '{query}' is not "
                            f"sufficiently related to the research topic "
                            f"(similarity={relevance:.2f}, threshold={RELEVANCE_THRESHOLD}). "
                            "Please rephrase the query so it stays on topic."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }

        for past_q in past_queries:
            sim = self._cosine_sim(query, past_q)
            if sim > REDUNDANCY_THRESHOLD:
                print(
                    f"[WebSearchGuard] BLOCKED: redundant "
                    f"('{query}' ≈ '{past_q}', sim={sim:.3f} > threshold={REDUNDANCY_THRESHOLD})"
                )
                return True, {
                    "messages": [
                        ToolMessage(
                            content=(
                                f"[SEARCH BLOCKED: Redundant] The query '{query}' is too "
                                f"similar to a previous search '{past_q}' "
                                f"(similarity={sim:.2f}, threshold={REDUNDANCY_THRESHOLD}). "
                                "Please use a different query that explores a new angle."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }

        print(f"[WebSearchGuard] Query approved.")
        return False, {"past_search_queries": past_queries + [query]}
