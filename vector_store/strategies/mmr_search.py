"""
Maximal Marginal Relevance (MMR) search strategy.
Balances relevance and diversity in results.
"""
import math
from typing import List, Optional

from ..core import SearchStrategy, VectorIndex, SearchQuery, SearchResult, VectorDocument


class MMRSearch(SearchStrategy):
    """
    Maximal Marginal Relevance search.
    
    MMR selects documents that are:
    1. Similar to the query (relevance)
    2. Dissimilar to already selected documents (diversity)
    
    Formula:
    MMR = argmax [ λ * sim(query, doc) - (1-λ) * max(sim(selected, doc)) ]
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR search.
        
        Args:
            lambda_param: Balance between relevance (1) and diversity (0).
                         Higher = more relevant, Lower = more diverse.
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
        self._lambda = lambda_param
    
    def search(
        self,
        index: VectorIndex,
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Execute MMR search."""
        # Get more candidates than needed for MMR selection
        candidate_count = query.top_k * 5
        candidate_query = SearchQuery(
            vector=query.vector,
            top_k=candidate_count,
            filter_fn=query.filter_fn,
            include_vectors=True,  # Need vectors for MMR
        )
        
        candidates = index.search(candidate_query)
        
        if not candidates:
            return []
        
        # MMR selection
        selected: List[SearchResult] = []
        remaining = candidates.copy()
        
        while len(selected) < query.top_k and remaining:
            best_mmr_score = float('-inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate MMR score
                mmr_score = self._calculate_mmr(
                    query_vector=query.vector,
                    selected=selected,
                    candidate=candidate,
                )
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
            
            # Select best candidate
            best = remaining.pop(best_idx)
            best.score = best_mmr_score  # Use MMR score as final score
            selected.append(best)
        
        # Re-rank
        for i, result in enumerate(selected):
            result.rank = i + 1
        
        # Remove vectors if not requested
        if not query.include_vectors:
            for result in selected:
                result.document.vector = []
        
        return selected
    
    def _calculate_mmr(
        self,
        query_vector: List[float],
        selected: List[SearchResult],
        candidate: SearchResult,
    ) -> float:
        """Calculate MMR score for a candidate."""
        # Similarity to query
        query_sim = candidate.score
        
        # Max similarity to already selected documents
        max_selected_sim = 0.0
        if selected:
            for sel in selected:
                sim = self._cosine_similarity(
                    sel.document.vector or sel.document.vector,
                    candidate.document.vector,
                )
                max_selected_sim = max(max_selected_sim, sim)
        
        # MMR formula
        mmr = self._lambda * query_sim - (1 - self._lambda) * max_selected_sim
        
        return mmr
    
    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        if not v1 or not v2:
            return 0.0
        
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
