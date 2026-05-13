"""End-of-session memory prompt helpers for BasicAgent."""

from datetime import date

from agents.utils.memory import ActiveMemory, StructuredMemory

from .feedback import FeedbackHandler


class BasicMemoryPromptBuilder:
    def _build_memory_carryover_note(self, current_date: date) -> str:
        next_active = self._get_next_active_date(current_date)
        next_text = (
            f"Next scheduled update: {next_active}."
            if next_active
            else "No later updates are scheduled."
        )
        return (
            "Your memory entries are the ONLY context retained between sessions. "
            f"You make updates every {self._get_timegap_days()} days. {next_text} "
            "In each session you get: search over news articles and the DataFrame "
            "(active question predictions, resolved question ground truths, your final "
            "predictions on resolved questions)."
        )

    def _build_resolution_recap_for_memory(self) -> str:
        """Build a compact recap of this session's resolutions for the memory update prompt."""
        feedback = getattr(self, '_last_feedback_data', None)
        if not feedback:
            return "No questions resolved this session.\n"
        resolved = feedback.get('resolved_today', [])
        if not resolved:
            return "No questions resolved this session.\n"

        lines = ["## QUESTIONS RESOLVED THIS SESSION (extract lessons from these)"]
        for item in resolved:
            dist = FeedbackHandler._format_distribution(
                item.get('my_pred_distribution') or {}
            )
            lines.append(
                f"- Q{item['qid']}: \"{item['title']}\"\n"
                f"  Predicted: {dist} | Truth: {item['ground_truth']} | Brier: {item['brier']:+.2f}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_memory_phase_prompt(self, current_date: date) -> str:
        """Build the memory-update prompt injected into the unified action loop."""
        if isinstance(self._memory, ActiveMemory):
            return self._build_active_memory_prompt(
                current_date, day_qids=getattr(self, '_day_qids', None)
            )
        if isinstance(self._memory, StructuredMemory):
            return self._build_structured_memory_prompt(current_date)
        return ""

    def _build_structured_memory_prompt(self, current_date: date) -> str:
        """Build the structured-memory update prompt."""
        memory_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if memory_index:
            index_block = f"Current memory index:\n{memory_index}\n"
        resolution_recap = self._build_resolution_recap_for_memory()

        prompt = f"""End of session {current_date}. Update your memory now.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

{resolution_recap}You currently have {self._memory.entry_count} memory entries (max {max_ent}).
{index_block}
### STEP 1: Extract lessons from resolved questions
For each question resolved this session, create a lesson entry:
- Name it `q<QID>-lesson-<topic>` (e.g., `q247-lesson-ceremony-precedent`)
- Content: What you predicted, what actually happened, WHY you were wrong/right, and a transferable rule for similar future questions
- Delete the old forecast entry for that QID — replace it with the lesson
- If no questions resolved this session, skip to Step 2.

Example lesson entry:
  name: q247-lesson-ceremony-precedent
  description: Lesson from Q247 resolution — weight historical precedent for rituals
  content: Predicted St. Peter's Square 0.70, Lateran 0.10. Truth: St. Peter's Square. Brier +0.85. Lesson: For ceremonial events with strong historical patterns (all recent popes used same venue), assign 0.85+ to the precedent option.

### STEP 2: Update meta-patterns
If you see a pattern across 2+ resolved questions (or resolved + active), add or update a meta-pattern entry:
- Name it `meta-<pattern-topic>`
- Content: The pattern, supporting evidence (which QIDs), and how to apply it

### STEP 3: Update active question entries
For questions you researched or updated today, store your current reasoning and key evidence. Merge duplicate entries about the same question.

### STEP 4: Cleanup
Delete stale entries (resolved questions with no useful lesson, outdated facts). Descriptions should answer: "Why would future-me read this?" — not just "What did I do today."

### Rules
- Every entry must contain a reusable insight or reasoning chain — NOT a log of what you did
- Do NOT just create entries like `2025-05-14-updates-summary` that just list probability changes. You must include a reusable insight or reasoning chain.
- Do NOT just store general forecasting advice or easily searchable facts. You must include a reusable insight or reasoning chain.

## RESPONSE FORMAT
Use the function tools from the tool schema. {'You may call multiple tools per turn for efficiency.' if self.config.parallel_tool_calls else 'Call one tool per turn.'}
Use `memory_retrieve`, `memory_new`, `memory_update`, and `memory_delete` for memory work.
{chr(10) + 'To avoid read/write conflicts, batch your calls by type:' + chr(10) + '- First turn(s): batch all `memory_retrieve` calls to review entries you need' + chr(10) + '- Next turn(s): batch all write operations (`memory_new`, `memory_update`, `memory_delete`) based on what you read' + chr(10) if self.config.parallel_tool_calls else ''}
You do NOT need to exhaust your remaining token budget — call `next_day()` as soon as your essential updates are complete.

Start with Step 1. If questions resolved this session, create lesson entries first."""
        return self._normalize_prompt_heading_spacing(prompt)

    def _build_plain_memory_prompt(self, current_date: date) -> str:
        """Build the plain-memory update prompt (full replacement text)."""
        prompt = f"""End of session {current_date}. You can now update your memory.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

Store things NOT recoverable from those tools:
1. Reasoning behind predictions and how you did on resolved questions that might help with unresolved questions — once a question resolves, your prediction remains visible in the dataframe, but your reasoning is never stored in the dataframe. Example: "Q149: PSG 0.70 because Sky Bet implied 55% and Inter eliminated in semis."
2. Performance patterns — track your accuracy across resolved questions so you can calibrate. Example: "Bookmaker odds were correct 80% across 15 sports questions; I should weight them more."
3. Non-obvious insights that search alone would not surface. Example: "'First country to X' questions almost always resolve to a major economy."
4. Critical hard-to-find facts directly relevant to active questions. Example: "ECB next meeting June 5 — relevant to Q72, Q108."

Do NOT store: general forecasting advice (already in your instructions), easily searchable facts, prediction outcomes without reasoning, or vague tracking lists without reasoning.
Aim to keep memory under 2000 characters. Prioritize recent and high-impact items and drop stale entries about resolved questions you have already learned from.

Respond with exactly one of:
- The full updated memory text only, with no extra wrapper tags or commentary.
- `NO_MEMORY_UPDATE` if you want to keep the current memory unchanged.

Current memory length: {len(self._memory)} characters"""
        return self._normalize_prompt_heading_spacing(prompt)

    def _build_active_memory_prompt(self, current_date: date, day_qids: set = None) -> str:
        """Build the active-memory update prompt."""
        mem_summary = self._memory.mem_summary(expanded_qids=day_qids)
        meta_index = self._memory.get_index()
        max_ent = self._memory._max_entries
        index_block = ""
        if meta_index:
            index_block = f"Current meta-insight index:\n{meta_index}\n"
        resolution_recap = self._build_resolution_recap_for_memory()

        prompt = f"""End of session {current_date}. Update your memory now.

## MEMORY UPDATE
{self._build_memory_carryover_note(current_date)}

{resolution_recap}Your memory has two layers, both retained between sessions. Everything else resets.

## Layer 1: QUESTION-SPECIFIC NOTES (mem_df: {self._memory.mem_count} entries)

Current entries:
{mem_summary}

In this, keep track of per-question reasoning, evidence, and calibration notes. Max 1000 chars per entry.
Only store what is NOT recoverable from the DataFrame or search.

## Layer 2: META-INSIGHTS ({self._memory.entry_count}/{max_ent} entries)

Use this to maintain cross-question patterns and calibration notes. NOT for question-specific reasoning (use mem_df for that).
{index_block}
### STEP 1: Extract lessons from resolved questions
For each question resolved this session:
- Create a meta-insight lesson entry (not mem_df — lessons are cross-question):
  Name it `q<QID>-lesson-<topic>` (e.g., `q247-lesson-ceremony-precedent`)
  Content: What you predicted, what actually happened, WHY you were wrong/right, and a transferable rule for similar future questions
- Delete the old mem_df entry for that QID using the `memory_delete` tool call - it's stale now
- If no questions resolved this session, skip to Step 2.

Example lesson entry:
  name: q247-lesson-ceremony-precedent
  description: Lesson from Q247 resolution — weight historical precedent for rituals
  content: Predicted St. Peter's Square 0.70, Lateran 0.10. Truth: St. Peter's Square. Brier +0.85. Lesson: For ceremonial events with strong historical patterns (all recent popes used same venue), assign 0.85+ to the precedent option.

### STEP 2: Update mem_df for questions you interacted with today
For each question you researched or forecasted today, add (using `memory_new`) or update (using `memory_update`) a mem_df entry with your current reasoning and key evidence.
- Include: your prediction, key evidence, sources, calibration notes
- Max 1000 chars — be concise but specific

### STEP 3: Update meta-patterns
If you see a pattern across 2+ resolved questions (or resolved + active), add or update a meta-insight entry:
- Name it `meta-<pattern-topic>`
- Content: The pattern, supporting evidence (which QIDs), and how to apply it

### STEP 4: Cleanup
Delete stale entries from both layers (resolved questions with no useful lesson, outdated facts, duplicates).
Descriptions should answer: "Why would future-me read this?" — not just "What did I do today."

### Rules
- Every entry must contain a reusable insight or reasoning chain — NOT a log of what you did
- Do NOT just create entries like `2025-05-14-updates-summary` that just list probability changes


## RESPONSE FORMAT
You will get to take multiple turns/actions to reason and make your updates.
Use the function tools from the tool schema. {'You may call multiple tools per turn for efficiency.' if self.config.parallel_tool_calls else 'Call one tool per turn.'}
Use `mem_add`, `mem_update`, and `mem_delete` for question-specific notes.
Use `memory_retrieve`, `memory_new`, `memory_update`, and `memory_delete` for meta-insights.
{chr(10) + 'To avoid read/write conflicts, batch your calls by type:' + chr(10) + '- First turn(s): batch all `memory_retrieve` calls to review entries you need' + chr(10) + '- Next turn(s): batch all write operations based on what you read' + chr(10) if self.config.parallel_tool_calls else ''}
You do NOT need to exhaust your remaining token budget — call `next_day()` as soon as your essential updates are complete.

Start with Step 1. If questions resolved this session, create lesson entries first.

Begin."""
        return self._normalize_prompt_heading_spacing(prompt)
