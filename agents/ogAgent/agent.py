from __future__ import annotations

from datetime import date
import re

from agents.allQAgent.agent import AllQAgent


class OgAgent(AllQAgent):
    """
    OgAgent: like AllQAgent warmup (Day 0 sweep), but uses the upstream dataset's
    original question prompt when available (e.g. OpenForesight HF field "prompt").

    We intentionally override ONLY the scoring + output-format instructions to match
    this codebase's submit format and multiclass Brier scoring.
    """

    def _build_warmup_system_prompt(self, current_date: date, q, forecast_interface=None) -> str:
        base_prompt = (getattr(q, "prompt", None) or "").strip()
        if not base_prompt:
            # Fallback: behave exactly like AllQAgent if dataset doesn't provide prompt.
            return super()._build_warmup_system_prompt(current_date, q, forecast_interface=forecast_interface)

        return self._surgically_patch_upstream_prompt(base_prompt, current_date, q)

    def _surgically_patch_upstream_prompt(self, base_prompt: str, current_date: date, q) -> str:
        """
        OpenForesight prompts end with a standard block that asks for:
          - <answer>...</answer>
          - <probability>...</probability>
        and describes a *binary* Brier-style rule for correctness.

        Our sim expects action XML (<action type="...">) and a *multiclass* probability
        distribution over outcomes (Brier over {named outcomes} U {truth}).

        We do a surgical replacement: swap the trailing scoring/output-format block
        in-place, leaving the rest of the upstream prompt intact.
        """
        # Observed invariant in the dataset: this exact lead-in appears in every prompt.
        marker = re.search(
            r"\n\s*Think step by step about the information provided,.*$",
            base_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # If the dataset format changes, fall back to appending (but keep behavior usable).
        if not marker:
            patched = base_prompt.rstrip()
            patched += "\n\n" + self._replacement_tail(current_date, q)
            return patched

        head = base_prompt[: marker.start()].rstrip()
        return head + "\n\n" + self._replacement_tail(current_date, q)

    def _replacement_tail(self, current_date: date, q) -> str:
        """Replacement for the upstream output-format + scoring tail block."""
        if getattr(self.config, "singleans", False):
            return """Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.""".rstrip()

        return f"""## SCORING (Multiclass Brier)
You are evaluated on **Brier Skill Score** = 1 - sum_i (p_i - y_i)^2 summed over **{{named outcomes}} U {{truth}}**.
- p_i = your probability for outcome i (0 if you didn't name it)
- y_i = 1 for the (matched) TRUE outcome, 0 otherwise
- If the TRUE outcome is not among your named outcomes, it is included with p=0.

Key mechanics:
- Submit a probability distribution over specific outcomes.
- You can submit at most {self.config.max_outcomes_per_question} outcomes.
- Do NOT use placeholders like "Unknown", "TBD", "Other", "N/A".
- Probabilities must sum to <= 1.0.

## OUTPUT FORMAT

### 1. Submit Forecast
<action type="submit">
<forecast qid="{q.qid}">
  <outcome name="Answer1" prob="0.50"/>
  <outcome name="Answer2" prob="0.30"/>
</forecast>
</action>

## RESPONSE FORMAT
Your response MUST be:
<reasoning>
...
</reasoning>
The <reasoning> section should contain your step-by-step thinking about the provided information and uncertainty.
<action type="submit">
...
</action>
""".rstrip()
