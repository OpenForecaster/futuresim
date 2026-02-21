"""
Answer matching for free-form forecasting outcomes.
Uses LLM for asymmetric semantic matching (predicted -> ground_truth).

Key rule: More specific predictions match less specific ground truth,
but vaguer predictions do NOT match more specific ground truth.
"""

from typing import List, Dict, Optional, Callable
import re
import time


def build_is_equivalent_prompt(predicted: str, ground_truth: str, question_title: str = None) -> str:
    """Build the is_equivalent/check_guess matcher prompt."""
    return f"""You are an objective judge of forecasting predictions.

Question: "{question_title}"
Predicted outcome: "{predicted}"
Ground truth (actual answer): "{ground_truth}"

Does the predicted outcome match the ground truth? Rules:
- YES if predictions are semantically equivalent (same meaning, different wording)
- YES if predicted outcome is MORE SPECIFIC than ground truth (e.g. "David Raya" matches "Raya")
- NO if predicted outcome contains generic text like "Unknown" or "Answer 1" or "Option 1"
- NO if predicted outcome is VAGUER/MORE GENERAL than ground truth (e.g., "a goalkeeper" does NOT match "David Raya")
- NO if they refer to different things

Essentially, you have to grade whether the forecaster correctly predicted the ground truth answer for the question.
Answer strictly "Yes" or "No"."""


def parse_is_equivalent_response(response: str) -> bool:
    """Parse matcher text output into a boolean equivalence decision."""
    return "yes" in (response or "").lower()


def build_find_match_prompt(candidate: str, existing_outcomes: List[str], question_title: str = None) -> str:
    """Build the find_match matcher prompt."""
    outcomes_list = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(existing_outcomes))
    question_line = f'Question: "{question_title}"\n\n' if question_title else ""
    return f"""You are an objective judge of forecasting predictions.

{question_line}New prediction: "{candidate}"

Existing predictions:
{outcomes_list}

Does the new prediction match any of the existing predictions semantically?
- Match if they mean the same thing or if new prediction is more specific
- Do NOT match if new prediction is vaguer/more general

If yes, respond with ONLY the number (e.g., "1" or "3").
If no match exists, respond with "None".

Answer:"""


def parse_find_match_response(response_text: str, existing_outcomes: List[str]) -> Optional[str]:
    """Parse matcher text output for find_match into a selected outcome or None."""
    response = (response_text or "").strip()
    if response.lower() == "none":
        return None

    numbers = re.findall(r"\d+", response)
    if not numbers:
        return None

    idx = int(numbers[0]) - 1
    if 0 <= idx < len(existing_outcomes):
        return existing_outcomes[idx]
    return None


class AnswerMatcher:
    """
    Semantic matching of prediction outcomes using LLM.
    
    Asymmetric matching: checks if a predicted outcome correctly
    matches the ground truth, allowing more specific predictions
    but rejecting vaguer ones.
    """
    
    def __init__(
        self,
        inference_provider,
        logger=None,
        cache_path: str = None,
        timing_callback: Optional[Callable[[float, float], None]] = None
    ):
        """
        Args:
            inference_provider: Object with chat(messages, sampling_params) method.
            logger: Optional SimLogger for logging matcher decisions.
            cache_path: Optional path to save/load persistent cache (JSON).
            timing_callback: Optional callback(duration_seconds, cost_usd) for matcher latency/cost.
        """
        self.inference = inference_provider
        self.logger = logger
        self.cache_path = cache_path
        self._timing_callback = timing_callback
        self._timing_count = 0
        self._timing_total_seconds = 0.0
        
        # Simple cache: (predicted, ground_truth, qid) -> bool
        self._cache: Dict[tuple, bool] = {}
        
        if cache_path:
            self.load_cache(cache_path)

    def _record_timing(self, duration: float, cost: float = 0) -> None:
        """Record matcher call timing and forward to optional callback."""
        self._timing_count += 1
        self._timing_total_seconds += duration
        if self._timing_callback:
            try:
                self._timing_callback(duration, cost)
            except Exception:
                # Timing callback should never break matcher correctness.
                pass
    
    def load_cache(self, path: str):
        """Load cache from a JSON file."""
        import os, json
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    raw_cache = json.load(f)
                    # Keys were strings in JSON, convert back to tuples
                    for k, v in raw_cache.items():
                        parts = k.split("|||")
                        if len(parts) == 3:
                            self._cache[tuple(parts)] = v
                if self.logger:
                    print(f"  Loaded matcher cache: {len(self._cache)} entries from {path}")
            except Exception as e:
                print(f"  Error loading matcher cache: {e}")

    def save_cache(self):
        """Save current cache to JSON file."""
        if not self.cache_path:
            return
            
        import json
        try:
            # Convert tuple keys to strings for JSON
            raw_cache = {f"{k[0]}|||{k[1]}|||{k[2]}": v for k, v in self._cache.items()}
            with open(self.cache_path, 'w') as f:
                json.dump(raw_cache, f)
        except Exception as e:
            print(f"  Error saving matcher cache: {e}")

    def _normalize(self, outcome: str) -> str:
        """Normalize outcome for exact match comparison."""
        return outcome.strip().lower()
    
    def is_equivalent(self, predicted: str, ground_truth: str, 
                      question_id: str = None, question_title: str = None,
                      match_type: str = "is_equivalent") -> bool:
        """
        Check if two outcome strings are semantically equivalent.
        """
        pred_norm = self._normalize(predicted)
        truth_norm = self._normalize(ground_truth)
        
        # Exact match
        if pred_norm == truth_norm:
            return True
        
        # Check cache
        cache_key = (pred_norm, truth_norm, str(question_id) if question_id else "None")
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Ask LLM
        result = self._llm_is_equivalent(predicted, ground_truth, question_id, question_title, match_type)
        self._cache[cache_key] = result
        
        # Save on update
        if self.cache_path:
            self.save_cache()
            
        return result
    
    def _llm_is_equivalent(self, predicted: str, ground_truth: str,
                           question_id: str = None, question_title: str = None,
                           match_type: str = "is_equivalent") -> bool:
        """Query LLM for semantic equivalence."""
        prompt = build_is_equivalent_prompt(
            predicted=predicted,
            ground_truth=ground_truth,
            question_title=question_title,
        )

        messages = [{"role": "user", "content": prompt}]
        usage = {}
        started = time.perf_counter()
        duration = 0.0
        try:
            response, usage = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        finally:
            duration = time.perf_counter() - started
            self._record_timing(duration, usage.get("cost", 0))
        
        is_equiv = parse_is_equivalent_response(response)
        
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
                output_data={"response": response, "is_equivalent": is_equiv},
                metadata={"duration_seconds": round(duration, 4)}
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
        prompt = build_find_match_prompt(
            candidate=candidate,
            existing_outcomes=existing_outcomes,
            question_title=question_title,
        )

        messages = [{"role": "user", "content": prompt}]
        usage = {}
        started = time.perf_counter()
        duration = 0.0
        try:
            response_text, usage = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 10})
        finally:
            duration = time.perf_counter() - started
            self._record_timing(duration, usage.get("cost", 0))
        match_result = parse_find_match_response(response_text, existing_outcomes)
        
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
                output_data={"response": response_text, "matched": match_result},
                metadata={"duration_seconds": round(duration, 4)}
            )
        
        return match_result
    
    def get_stats(self) -> Dict:
        """Get statistics about answer matching."""
        snapshot = self.get_timing_snapshot()
        count = snapshot["matcher_count"]
        total = snapshot["matcher_total_seconds"]
        return {
            "cache_size": len(self._cache),
            "matcher_count": count,
            "matcher_total_seconds": round(total, 3),
            "matcher_avg_seconds": round(
                total / count, 3
            ) if count > 0 else 0.0,
        }

    def get_timing_snapshot(self) -> Dict[str, float]:
        """
        Return raw matcher timing counters (no rounding).

        Useful for computing per-day deltas from cumulative matcher stats.
        """
        return {
            "matcher_count": int(self._timing_count),
            "matcher_total_seconds": float(self._timing_total_seconds),
        }
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
