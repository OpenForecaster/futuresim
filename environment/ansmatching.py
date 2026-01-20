"""
Answer matching for free-form forecasting outcomes.
Uses LLM for semantic equivalence with Union-Find for transitivity.
"""

from typing import List, Dict, Tuple, Optional, Set
import re


class AnswerMatcher:
    """
    Semantic matching of prediction outcomes using LLM.
    
    Uses Union-Find to maintain transitive equivalence classes.
    Tracks inconsistencies when LLM contradicts transitivity.
    """
    
    def __init__(self, inference_provider, logger=None):
        """
        Args:
            inference_provider: Object with chat(messages, sampling_params) method.
            logger: Optional SimLogger for logging matcher decisions.
        """
        self.inference = inference_provider
        self.logger = logger
        
        # Union-Find: normalized outcome -> parent
        self._parent: Dict[str, str] = {}
        
        # Cache of LLM responses: (a, b) sorted -> equivalent?
        self._llm_cache: Dict[Tuple[str, str], bool] = {}
        
        # Track all equivalences for debugging
        self._all_queries: List[Tuple[str, str, bool]] = []
        
        # Detected inconsistencies: (a, b, explanation)
        self._inconsistencies: List[Tuple[str, str, str]] = []
    
    def _normalize(self, outcome: str) -> str:
        """Normalize outcome for comparison."""
        return outcome.strip().lower()
    
    def _find(self, outcome: str) -> str:
        """Find canonical representative with path compression."""
        norm = self._normalize(outcome)
        if norm not in self._parent:
            self._parent[norm] = norm
            return norm
        
        # Path compression
        if self._parent[norm] != norm:
            self._parent[norm] = self._find(self._parent[norm])
        return self._parent[norm]
    
    def _union(self, a: str, b: str):
        """Merge equivalence classes of a and b."""
        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            # Always make lexicographically smaller one the parent (deterministic)
            if root_a < root_b:
                self._parent[root_b] = root_a
            else:
                self._parent[root_a] = root_b
    
    def _get_class_members(self, outcome: str) -> Set[str]:
        """Get all members of the equivalence class containing outcome."""
        root = self._find(outcome)
        return {k for k in self._parent if self._find(k) == root}
    
    def _check_consistency(self, a: str, b: str) -> Optional[str]:
        """
        Check if merging a and b would violate transitivity.
        Returns explanation string if inconsistency found, None otherwise.
        """
        class_a = self._get_class_members(a)
        class_b = self._get_class_members(b)
        
        # Check if any pair across classes was marked as NOT equivalent
        for x in class_a:
            for y in class_b:
                cache_key = (x, y) if x < y else (y, x)
                if cache_key in self._llm_cache and not self._llm_cache[cache_key]:
                    return f"LLM said '{x}' ≠ '{y}', but now '{a}' ≈ '{b}' implies they should be equal"
        
        return None
        
    def is_equivalent(self, outcome_a: str, outcome_b: str) -> bool:
        """
        Check if two outcomes are semantically equivalent.
        Uses Union-Find to enforce transitivity.
        """
        a_norm = self._normalize(outcome_a)
        b_norm = self._normalize(outcome_b)
        
        # Exact match
        if a_norm == b_norm:
            return True
        
        # Already in same equivalence class (transitively equivalent)
        if self._find(a_norm) == self._find(b_norm):
            return True
        
        # Check cache
        cache_key = (a_norm, b_norm) if a_norm < b_norm else (b_norm, a_norm)
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Ask LLM
        result = self._llm_is_equivalent(outcome_a, outcome_b)
        self._llm_cache[cache_key] = result
        self._all_queries.append((outcome_a, outcome_b, result))
        
        if result:
            # Check for inconsistency before merging
            inconsistency = self._check_consistency(a_norm, b_norm)
            if inconsistency:
                self._inconsistencies.append((outcome_a, outcome_b, inconsistency))
                print(f"⚠️ Transitivity inconsistency: {inconsistency}")
            
            # Merge equivalence classes
            self._union(a_norm, b_norm)
        
        return result
    
    def _llm_is_equivalent(self, outcome_a: str, outcome_b: str) -> bool:
        """Query LLM for semantic equivalence."""
        prompt = f"""You are an objective judge of forecasting predictions.
Outcome A: "{outcome_a}"
Outcome B: "{outcome_b}"

Are these two outcomes semantically equivalent? i.e. do they represent the exact same result?
Answer strictly with "Yes" or "No"."""

        messages = [{"role": "user", "content": prompt}]
        response, _ = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        
        is_equiv = "yes" in response.lower()
        
        if self.logger:
            self.logger.log_matcher(
                input_data={"type": "is_equivalent", "outcome_a": outcome_a, "outcome_b": outcome_b},
                output_data={"response": response, "is_equivalent": is_equiv}
            )
            
        return is_equiv

    def find_match(self, candidate: str, existing_outcomes: List[str]) -> Optional[str]:
        """
        Find if candidate matches any existing outcome.
        Uses Union-Find for known equivalences, LLM for new comparisons.
        Returns the matching existing outcome string, or None.
        """
        if not existing_outcomes:
            return None
        
        candidate_norm = self._normalize(candidate)
        
        # Check Union-Find first (known equivalences)
        for existing in existing_outcomes:
            existing_norm = self._normalize(existing)
            if self._find(candidate_norm) == self._find(existing_norm):
                return existing
        
        # Quick exact match
        for existing in existing_outcomes:
            if self._normalize(existing) == candidate_norm:
                return existing
        
        # Single LLM query for batch comparison
        if len(existing_outcomes) == 1:
            if self.is_equivalent(candidate, existing_outcomes[0]):
                return existing_outcomes[0]
            return None
        
        # Batch query for multiple outcomes
        outcomes_list = "\n".join(f"{i+1}. {o}" for i, o in enumerate(existing_outcomes))
        
        prompt = f"""You are an objective judge of forecasting predictions.

New prediction: "{candidate}"

Existing predictions:
{outcomes_list}

Does the new prediction match any of the existing predictions semantically?
If yes, respond with ONLY the number of the matching prediction (e.g., "1" or "3").
If no match exists, respond with "None".

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        response_text, _ = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        response = response_text.strip()
        
        match_result = None
        if response.lower() != "none":
            try:
                numbers = re.findall(r'\d+', response)
                if numbers:
                    idx = int(numbers[0]) - 1
                    if 0 <= idx < len(existing_outcomes):
                        match_result = existing_outcomes[idx]
            except Exception:
                pass
        
        if self.logger:
            self.logger.log_matcher(
                input_data={"type": "find_match", "candidate": candidate, "existing": existing_outcomes},
                output_data={"response": response_text, "matched": match_result}
            )
        
        if response.lower() == "none":
            return None
        
        try:
            numbers = re.findall(r'\d+', response)
            if numbers:
                idx = int(numbers[0]) - 1
                if 0 <= idx < len(existing_outcomes):
                    matched = existing_outcomes[idx]
                    # Record this equivalence in Union-Find
                    self._union(candidate_norm, self._normalize(matched))
                    self._all_queries.append((candidate, matched, True))
                    return matched
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_canonical(self, outcome: str) -> str:
        """Get the canonical representative for an outcome."""
        return self._find(outcome)
    
    def get_equivalence_class(self, outcome: str) -> Set[str]:
        """Get all outcomes equivalent to the given one."""
        return self._get_class_members(outcome)
    
    def get_inconsistencies(self) -> List[Tuple[str, str, str]]:
        """Get all detected transitivity inconsistencies."""
        return self._inconsistencies.copy()
    
    def get_stats(self) -> Dict:
        """Get statistics about answer matching."""
        return {
            "total_queries": len(self._all_queries),
            "cache_size": len(self._llm_cache),
            "equivalence_classes": len(set(self._find(k) for k in self._parent)),
            "total_outcomes": len(self._parent),
            "inconsistencies": len(self._inconsistencies)
        }
    
    def clear(self):
        """Clear all state."""
        self._parent.clear()
        self._llm_cache.clear()
        self._all_queries.clear()
        self._inconsistencies.clear()
