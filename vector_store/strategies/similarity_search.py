"""
Similarity search strategy.
Standard nearest neighbor search.
"""
from typing import List

from ..core import SearchStrategy, VectorIndex, SearchQuery, SearchResult


class SimilaritySearch(SearchStrategy):
    """
    Standard similarity search strategy.
    Returns top-k most similar documents.
    """
    
    def __init__(self, normalize_scores: bool = False):
        """
        Initialize strategy.
        
        Args:
            normalize_scores: Whether to normalize scores to [0, 1].
        """
        self._normalize_scores = normalize_scores
    
    def search(
        self,
        index: VectorIndex,
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Execute similarity search."""
        results = index.search(query)
        
        if self._normalize_scores and results:
            # Normalize scores to [0, 1] using min-max
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for result in results:
                    result.score = (result.score - min_score) / score_range
            else:
                for result in results:
                    result.score = 1.0 if result.score == max_score else 0.0
        
        return results
