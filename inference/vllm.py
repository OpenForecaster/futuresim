from typing import Dict, Any, List, Optional, Tuple
import os

# Global singleton storage to keep models loaded across agent instances
_VLLM_ENGINES = {}

class VLLMInference:
    def __init__(self, model_path: str, model_name: str = None, 
                 max_model_len: int = 32768, **kwargs):
        """
        Initialize VLLM engine. 
        If model_path is already loaded, reuse the engine.
        
        Args:
            model_path: Path to the model
            model_name: Optional display name
            max_model_len: Maximum context length (default 32768)
            **kwargs: Additional args passed to vllm.LLM
        """
        global _VLLM_ENGINES
        self.model_path = model_path
        self.model_name = model_name or model_path
        
        # Include max_model_len in cache key since different lengths need different engines
        cache_key = f"{model_path}::{max_model_len}"
        
        if cache_key in _VLLM_ENGINES:
            self.llm = _VLLM_ENGINES[cache_key]
        else:
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError("vllm module not found. Please install vllm.")
                
            if "trust_remote_code" not in kwargs:
                kwargs["trust_remote_code"] = True
            
            # Set max_model_len to prevent OOM on limited GPU memory
            self.llm = LLM(model=model_path, max_model_len=max_model_len, **kwargs)
            _VLLM_ENGINES[cache_key] = self.llm
            
    def chat(self, messages: List[Dict[str, str]], sampling_params: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Chat completion using messages format.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            sampling_params: Dict with temperature, max_tokens, etc.
            
        Returns:
            The assistant's response text.
        """
        from vllm import SamplingParams
        
        sp = SamplingParams(**sampling_params)
        
        # vLLM's chat() expects a list of conversations (batch of 1 here)
        outputs = self.llm.chat([messages], sp, use_tqdm=False)
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        usage = {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
        }
        
        return generated_text, usage

