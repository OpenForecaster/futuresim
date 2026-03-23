
- should we have written in tinker interface and used skyrl-tx instead of the default skyrl environment?

- skyrl actual tool format / tokens is misaligned with qwen3.5 recommendations we need to fix this allow default handling fo the qwen3.5 tokenization format. 

- current skyrl loop basically re-codes environment step logic. this can cause misalginment with eval environment logic. why cant we just inherit run qwen action loop? ig one reason is because we only want warmup phase, dont have memory etc. still then qwenagent action loop should be modular enough to support inheriting without them or passing args that disable them while we continue to use it. basically ensure that happens rather than having to re-implement the codeflow as in the future when we want to extend the skyrl training environment just using existing eval agent action loop will make life easy (just need to pass more through config params through skyrl then)

- convert from warmup to questions come at some start date (can fix to 2 months before resolution or sth when we dont have a clear start date) and when they come then we run a separate question-specific warmup, just to keep things "realistic".