"""
Answer matching for free-form forecasting outcomes.
Uses LLM for asymmetric semantic matching (predicted -> ground_truth).

Key rule: More specific predictions match less specific ground truth,
but vaguer predictions do NOT match more specific ground truth.
"""

from typing import List, Dict, Optional
import re


class AnswerMatcher:
    """
    Semantic matching of prediction outcomes using LLM.
    
    Asymmetric matching: checks if a predicted outcome correctly
    matches the ground truth, allowing more specific predictions
    but rejecting vaguer ones.
    """
    
    def __init__(self, inference_provider, logger=None):
        """
        Args:
            inference_provider: Object with chat(messages, sampling_params) method.
            logger: Optional SimLogger for logging matcher decisions.
        """
        self.inference = inference_provider
        self.logger = logger
        
        # Simple cache: (predicted, ground_truth, qid) -> bool
        self._cache: Dict[tuple, bool] = {}
    
    def _normalize(self, outcome: str) -> str:
        """Normalize outcome for exact match comparison."""
        return outcome.strip().lower()
    
    def is_equivalent(self, predicted: str, ground_truth: str, 
                      question_id: str = None, question_title: str = None,
                      match_type: str = "is_equivalent") -> bool:
        """
        Check if two outcome strings are semantically equivalent.
        
        Args:
            predicted: The predicted outcome string
            ground_truth: The ground truth string (or another outcome to compare)
            question_id: Optional question ID for logging context
            question_title: Optional question title for logging context
            match_type: Type of match operation ("check_guess", "expand_set", "is_equivalent")
        """
        pred_norm = self._normalize(predicted)
        truth_norm = self._normalize(ground_truth)
        
        # Exact match
        if pred_norm == truth_norm:
            return True
        
        # Check cache
        cache_key = (pred_norm, truth_norm, question_id)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Ask LLM
        result = self._llm_is_equivalent(predicted, ground_truth, question_id, question_title, match_type)
        self._cache[cache_key] = result
        
        return result
    
    def _llm_is_equivalent(self, predicted: str, ground_truth: str,
                           question_id: str = None, question_title: str = None,
                           match_type: str = "is_equivalent") -> bool:
        """Query LLM for semantic equivalence."""
        prompt = f"""You are an objective judge of forecasting predictions.

Question: "{question_title}"
Predicted outcome: "{predicted}"
Ground truth (actual answer): "{ground_truth}"

Does the predicted outcome match the ground truth? Rules:
- YES if predictions are semantically equivalent (same meaning, different wording)
- YES if predicted outcome is MORE SPECIFIC than ground truth (e.g., "David Raya" matches "Raya")
- NO if predicted outcome is VAGUER/MORE GENERAL than ground truth (e.g., "a goalkeeper" does NOT match "David Raya")
- NO if they refer to different things

Answer strictly "Yes" or "No"."""

        messages = [{"role": "user", "content": prompt}]
        response, _ = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        
        is_equiv = "yes" in response.lower()
        
        if self.logger:
            input_data = {
                "type": match_type, 
                "predicted": predicted, 
                "ground_truth": ground_truth
            }
            if question_id:
                input_data["question_id"] = question_id
            if question_title:
                input_data["question_title"] = question_title
            self.logger.log_matcher(
                input_data=input_data,
                output_data={"response": response, "is_equivalent": is_equiv}
            )
            
        return is_equiv

    def find_match(self, candidate: str, existing_outcomes: List[str],
                   question_id: str = None, question_title: str = None) -> Optional[str]:
        """
        Find if candidate matches any existing outcome.
        
        Args:
            candidate: New prediction to match
            existing_outcomes: List of existing outcomes to compare against
            question_id: Optional question ID for logging
            question_title: Optional question title for context
            
        Returns:
            The matching existing outcome string, or None
        """
        if not existing_outcomes:
            return None
        
        candidate_norm = self._normalize(candidate)
        
        # Quick exact match
        for existing in existing_outcomes:
            if self._normalize(existing) == candidate_norm:
                return existing
        
        # Single outcome - use is_equivalent
        if len(existing_outcomes) == 1:
            if self.is_equivalent(candidate, existing_outcomes[0],
                                  question_id=question_id, question_title=question_title,
                                  match_type="expand_set"):
                return existing_outcomes[0]
            return None
        
        # Batch query for multiple outcomes
        outcomes_list = "\n".join(f"{i+1}. {o}" for i, o in enumerate(existing_outcomes))
        question_line = f'Question: "{question_title}"\n\n' if question_title else ""
        
        prompt = f"""You are an objective judge of forecasting predictions.

{question_line}New prediction: "{candidate}"

Existing predictions:
{outcomes_list}

Does the new prediction match any of the existing predictions semantically?
- Match if they mean the same thing or if new prediction is more specific
- Do NOT match if new prediction is vaguer/more general

If yes, respond with ONLY the number (e.g., "1" or "3").
If no match exists, respond with "None".

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        response_text, _ = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        response = response_text.strip()
        
        # Parse result
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
        
        # Log
        if self.logger:
            input_data = {
                "type": "find_match", 
                "candidate": candidate, 
                "existing": existing_outcomes
            }
            if question_id:
                input_data["question_id"] = question_id
            if question_title:
                input_data["question_title"] = question_title
            self.logger.log_matcher(
                input_data=input_data,
                output_data={"response": response_text, "matched": match_result}
            )
        
        return match_result
    
    def get_stats(self) -> Dict:
        """Get statistics about answer matching."""
        return {
            "cache_size": len(self._cache),
        }
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
