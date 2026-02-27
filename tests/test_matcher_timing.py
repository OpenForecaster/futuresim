from agents.utils.timing import AgentTimer
from environment.ansmatching import AnswerMatcher


class DummyInference:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, sampling_params):
        del messages, sampling_params
        if self.calls >= len(self._responses):
            raise RuntimeError("No dummy response left")
        response = self._responses[self.calls]
        self.calls += 1
        return response, {}


def test_agent_timer_records_matcher_category():
    timer = AgentTimer()
    timer.record("matcher", 0.123)

    summary = timer.get_summary()
    assert summary["matcher_count"] == 1
    assert summary["matcher_total_seconds"] == 0.123
    assert summary["matcher_avg_seconds"] == 0.123


def test_matcher_timing_callback_called_only_for_uncached_llm_matches():
    durations = []
    inference = DummyInference(["Yes"])
    matcher = AnswerMatcher(
        inference,
        timing_callback=lambda duration, cost=0: durations.append(duration),
    )

    assert matcher.is_equivalent("alpha", "beta") is True
    assert inference.calls == 1
    assert len(durations) == 1
    assert durations[0] >= 0.0

    # Cached; should not call LLM or timing callback again.
    assert matcher.is_equivalent("alpha", "beta") is True
    assert inference.calls == 1
    assert len(durations) == 1


def test_find_match_records_timing_for_batch_match_call():
    durations = []
    inference = DummyInference(["2"])
    matcher = AnswerMatcher(
        inference,
        timing_callback=lambda duration, cost=0: durations.append(duration),
    )

    match = matcher.find_match("candidate", ["first", "second"])
    assert match == "second"
    assert inference.calls == 1
    assert len(durations) == 1
    assert durations[0] >= 0.0
