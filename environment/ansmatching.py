"""
Answer matching for free-form forecasting outcomes.
Uses LLM for asymmetric semantic matching (predicted -> ground_truth).

Key rule: More specific predictions match less specific ground truth,
but vaguer predictions do NOT match more specific ground truth.

Supports async batch matching via ``batch_is_equivalent`` / ``warmup_cache``
to parallelise OpenRouter calls (up to ``max_concurrency`` at a time).
"""

from typing import List, Dict, Optional, Callable, Tuple
import asyncio
import atexit
import json
import os
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


API_URL = "https://openrouter.ai/api/v1/chat/completions"


async def _async_openrouter_chat(
    client,
    model: str,
    messages: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    temperature: float = 0.0,
    max_tokens: int = 50,
    max_retries: int = 3,
) -> Tuple[str, dict]:
    """Single async OpenRouter chat completion with retries and concurrency control.

    Uses a shared ``httpx.AsyncClient`` passed in by the caller.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }

    max_retries = min(3, max(1, int(max_retries)))
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.post(API_URL, json=payload)
            if resp.status_code == 429:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"] or ""
                return content, data.get("usage", {})
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

    return "", {}


async def async_batch_openrouter_chat(
    api_key: str,
    model: str,
    message_batches: List[List[Dict[str, str]]],
    max_concurrency: int = 300,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> List[Tuple[str, dict]]:
    """Fire many OpenRouter chat calls concurrently.

    Uses a single shared httpx.AsyncClient with a connection pool for all requests.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier string.
        message_batches: List of message lists (one per call).
        max_concurrency: Max concurrent requests (semaphore size).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        List of (response_text, usage_dict) in the same order as input.
    """
    import httpx

    sem = asyncio.Semaphore(max_concurrency)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )
    async with httpx.AsyncClient(
        timeout=120.0, headers=headers, limits=limits
    ) as client:
        tasks = [
            _async_openrouter_chat(client, model, msgs, sem, temperature, max_tokens)
            for msgs in message_batches
        ]
        return list(await asyncio.gather(*tasks))


def run_async(coro):
    """Run an async coroutine from sync code, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


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
        timing_callback: Optional[Callable[[float, float], None]] = None,
    ):
        """
        Args:
            inference_provider: Object with chat(messages, sampling_params) method.
            logger: Optional SimLogger for logging matcher decisions.
            cache_path: Optional path to load/save persistent cache (JSON). Loaded at init;
                merged writes happen only from :meth:`persist_cache` (e.g. process exit).
            timing_callback: Optional callback(duration_seconds, cost_usd) for matcher latency/cost.
        """
        self.inference = inference_provider
        self.logger = logger
        self.cache_path = cache_path
        self._timing_callback = timing_callback
        self._timing_count = 0
        self._timing_total_seconds = 0.0
        self._timing_total_cost = 0.0
        self._cache_dirty = False
        self._cache_hit_count = 0
        self._cache_miss_count = 0

        # (pred_norm, truth_norm, qid, title_norm) -> bool
        self._cache: Dict[tuple, bool] = {}

        if cache_path:
            self.load_cache(cache_path)
            atexit.register(self._atexit_persist_cache)

    def _record_timing(self, duration: float, cost: float = 0) -> None:
        """Record matcher call timing and forward to optional callback."""
        self._timing_count += 1
        self._timing_total_seconds += duration
        self._timing_total_cost += cost
        if self._timing_callback:
            try:
                self._timing_callback(duration, cost)
            except Exception:
                # Timing callback should never break matcher correctness.
                pass
    
    @staticmethod
    def _key_to_disk(k: tuple) -> str:
        """Stable JSON key for on-disk cache entries."""
        if len(k) == 3:
            k = (k[0], k[1], k[2], "")
        if len(k) != 4:
            raise ValueError(f"Invalid matcher cache key tuple: {k!r}")
        return json.dumps([k[0], k[1], k[2], k[3]], ensure_ascii=False, sort_keys=False)

    @staticmethod
    def _disk_to_key(s: str) -> tuple:
        try:
            parts = json.loads(s)
            if isinstance(parts, list) and len(parts) == 4:
                return tuple(str(x) for x in parts)
        except Exception:
            pass
        legacy = s.split("|||")
        if len(legacy) == 3:
            return (legacy[0], legacy[1], legacy[2], "")
        if len(legacy) == 4:
            return tuple(legacy)
        raise ValueError(f"Unrecognized matcher cache key: {s!r}")

    def load_cache(self, path: str):
        """Load cache from a JSON object mapping string keys -> bool."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_cache = json.load(f)
                for k, v in raw_cache.items():
                    key = self._disk_to_key(str(k))
                    self._cache[key] = bool(v)
                print(f"  Loaded matcher cache: {len(self._cache)} entries from {path}")
            except Exception as e:
                print(f"  Error loading matcher cache: {e}")

    def persist_cache(self) -> None:
        """Merge in-memory cache into the JSON file (atomic replace, POSIX file lock)."""
        if not self.cache_path:
            return
        path = str(self.cache_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lock_path = path + ".lock"
        tmp_path = f"{path}.{os.getpid()}.tmp"
        written = False

        try:
            import fcntl  # type: ignore

            with open(lock_path, "a+", encoding="utf-8") as lockf:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                try:
                    merged: Dict[str, bool] = {}
                    if os.path.exists(path):
                        try:
                            with open(path, "r", encoding="utf-8") as rf:
                                cur = json.load(rf)
                            if isinstance(cur, dict):
                                merged = {str(k): bool(v) for k, v in cur.items()}
                        except Exception:
                            merged = {}
                    for tup, val in list(self._cache.items()):
                        merged[self._key_to_disk(tup)] = bool(val)
                    payload = json.dumps(merged, ensure_ascii=False, sort_keys=True)
                    with open(tmp_path, "w", encoding="utf-8") as wf:
                        wf.write(payload)
                    os.replace(tmp_path, path)
                    written = True
                finally:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Non-POSIX or lock failure: best-effort single-writer replace.
            try:
                merged = {}
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as rf:
                            cur = json.load(rf)
                        if isinstance(cur, dict):
                            merged = {str(k): bool(v) for k, v in cur.items()}
                    except Exception:
                        merged = {}
                for tup, val in list(self._cache.items()):
                    merged[self._key_to_disk(tup)] = bool(val)
                payload = json.dumps(merged, ensure_ascii=False, sort_keys=True)
                with open(tmp_path, "w", encoding="utf-8") as wf:
                    wf.write(payload)
                os.replace(tmp_path, path)
                written = True
            except Exception as exc:
                print(f"  Error saving matcher cache: {exc}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if written:
            self._cache_dirty = False
            print(f"  Persisted matcher cache: {len(self._cache)} entries -> {path}")

    def save_cache(self):
        """Backward-compatible alias for :meth:`persist_cache`."""
        self.persist_cache()

    def _atexit_persist_cache(self) -> None:
        try:
            if self.cache_path and self._cache_dirty:
                self.persist_cache()
        except Exception:
            pass

    def _normalize(self, outcome: str) -> str:
        """Normalize outcome for exact match comparison."""
        return outcome.strip().lower()

    def _make_cache_key(
        self,
        pred_norm: str,
        truth_norm: str,
        question_id: Optional[str],
        question_title: Optional[str],
    ) -> tuple:
        return (
            pred_norm,
            truth_norm,
            str(question_id) if question_id else "None",
            self._normalize(question_title or ""),
        )

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

        cache_key = self._make_cache_key(pred_norm, truth_norm, question_id, question_title)
        if cache_key in self._cache:
            self._cache_hit_count += 1
            return self._cache[cache_key]

        legacy_key = (pred_norm, truth_norm, str(question_id) if question_id else "None")
        if legacy_key in self._cache:
            val = self._cache[legacy_key]
            self._cache[cache_key] = val
            self._cache_hit_count += 1
            return val

        # Ask LLM
        result = self._llm_is_equivalent(predicted, ground_truth, question_id, question_title, match_type)
        self._cache[cache_key] = result
        if self.cache_path:
            self._cache_dirty = True
        self._cache_miss_count += 1

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

    # ------------------------------------------------------------------
    # Async batch matching
    # ------------------------------------------------------------------

    def batch_is_equivalent(
        self,
        items: List[Tuple[str, str, Optional[str], Optional[str]]],
        max_concurrency: int = 300,
    ) -> List[bool]:
        """Fire many is_equivalent calls concurrently via async httpx.

        Each *item* is ``(predicted, ground_truth, question_id, question_title)``.
        Returns a list of booleans in the same order.

        Calls that are already cached or are exact matches skip the API.
        """
        results: List[Optional[bool]] = [None] * len(items)
        pending_indices: List[int] = []
        pending_meta: List[Tuple[str, str, str, str, tuple]] = []
        pending_messages: List[List[Dict[str, str]]] = []

        for i, (predicted, ground_truth, qid, qtitle) in enumerate(items):
            pred_norm = self._normalize(predicted)
            truth_norm = self._normalize(ground_truth)
            if pred_norm == truth_norm:
                results[i] = True
                continue
            cache_key = self._make_cache_key(pred_norm, truth_norm, qid, qtitle)
            if cache_key in self._cache:
                self._cache_hit_count += 1
                results[i] = self._cache[cache_key]
                continue
            legacy_key = (pred_norm, truth_norm, str(qid) if qid else "None")
            if legacy_key in self._cache:
                self._cache_hit_count += 1
                results[i] = self._cache[legacy_key]
                self._cache[cache_key] = self._cache[legacy_key]
                continue
            prompt = build_is_equivalent_prompt(predicted, ground_truth, qtitle)
            pending_indices.append(i)
            pending_meta.append((predicted, ground_truth, qid, qtitle, cache_key))
            pending_messages.append([{"role": "user", "content": prompt}])

        if not pending_messages:
            return [r for r in results]

        started = time.perf_counter()
        responses = run_async(async_batch_openrouter_chat(
            api_key=self._resolve_api_key(),
            model=self._resolve_model(),
            message_batches=pending_messages,
            max_concurrency=max_concurrency,
        ))
        batch_duration = time.perf_counter() - started

        per_call_duration = batch_duration / max(len(responses), 1)
        for j, (response_text, usage) in enumerate(responses):
            idx = pending_indices[j]
            predicted, ground_truth, qid, qtitle, cache_key = pending_meta[j]
            is_equiv = parse_is_equivalent_response(response_text)
            self._cache[cache_key] = is_equiv
            self._record_timing(per_call_duration, usage.get("cost", 0))
            results[idx] = is_equiv

            if self.logger:
                input_data = {
                    "type": "check_guess",
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                }
                if qid:
                    input_data["question_id"] = qid
                if qtitle:
                    input_data["question_title"] = qtitle
                self.logger.log_matcher(
                    input_data=input_data,
                    output_data={"response": response_text, "is_equivalent": is_equiv},
                    metadata={"duration_seconds": round(per_call_duration, 4), "batch": True},
                )

        if pending_messages and self.cache_path:
            self._cache_dirty = True
        self._cache_miss_count += len(pending_messages)

        return [r if r is not None else False for r in results]

    def warmup_cache(
        self,
        items: List[Tuple[str, str, Optional[str], Optional[str]]],
        max_concurrency: int = 300,
    ) -> None:
        """Pre-populate cache for a batch of (predicted, gt, qid, qtitle) tuples.

        After this call every subsequent ``is_equivalent`` for these pairs
        will be a cache hit (no API call).
        """
        if not items:
            return
        uncached = []
        for predicted, ground_truth, qid, qtitle in items:
            pred_norm = self._normalize(predicted)
            truth_norm = self._normalize(ground_truth)
            if pred_norm == truth_norm:
                continue
            cache_key = self._make_cache_key(pred_norm, truth_norm, qid, qtitle)
            legacy_key = (pred_norm, truth_norm, str(qid) if qid else "None")
            if cache_key not in self._cache and legacy_key not in self._cache:
                uncached.append((predicted, ground_truth, qid, qtitle))
        if not uncached:
            return
        print(f"  [Matcher] warming cache: {len(uncached)} calls (concurrency={max_concurrency})")
        started = time.perf_counter()
        self.batch_is_equivalent(uncached, max_concurrency=max_concurrency)
        elapsed = time.perf_counter() - started
        print(f"  [Matcher] cache warm done: {len(uncached)} calls in {elapsed:.1f}s "
              f"({len(uncached)/max(elapsed,0.01):.1f} calls/s)")

    def _resolve_api_key(self) -> str:
        """Extract the API key from the inference provider."""
        if hasattr(self.inference, 'api_key'):
            return self.inference.api_key
        return os.environ.get("OPENROUTER_API_KEY", "")

    def _resolve_model(self) -> str:
        """Extract the model name from the inference provider."""
        if hasattr(self.inference, 'model'):
            return self.inference.model
        return "deepseek/deepseek-v3.2"

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
            response_text, usage = self.inference.chat(messages, {"temperature": 0.0, "max_tokens": 50})
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
            "cache_hits": int(self._cache_hit_count),
            "cache_misses": int(self._cache_miss_count),
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
            "matcher_total_cost": float(self._timing_total_cost),
            "cache_hits": int(self._cache_hit_count),
            "cache_misses": int(self._cache_miss_count),
            "cache_size": int(len(self._cache)),
        }
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
