- qwenagent which shifts basicagent to native qwen3.5 agent tool call and prompt formats. eval on just warmup phase. https://github.com/QwenLM/Qwen-Agent https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html 

- test (warmup phase style 10 actions) upperbound of qwen3.5 agent both with 1 day before resolution of each q testing, and post-resolution fixed date across qs testing where acc should be really high as search should basically ideally surface the answer.

- check whether in subsequent day search queries model is even pulling up newer search articles or not by maybe computing mean article date or fraction of articles after day 0 pulled up in searches from subsequent days. 

- setup basic search agent (day 0 warmup style) grpo training in skyrl just to port the environment into skyrl and setup initial training, can extend to full simulation later. 

- convert from actions to tokens as the constraint.

- for denser feedback also add in the guardian validation set and start simulation then as it has more qs in the month of april itself which can be used to accumulate memory.

- convert from warmup to questions come at some start date (can fix to 2 months before resolution or sth when we dont have a clear start date) and when they come then we run a separate question-specific warmup, just to keep things "realistic".