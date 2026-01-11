from typing import Dict, Any, List, Optional
import os

# Global singleton storage to keep models loaded across agent instances
_VLLM_ENGINES = {}

class VLLMInference:
    def __init__(self, model_path: str, model_name: str = None, **kwargs):
        """
        Initialize VLLM engine. 
        If model_path is already loaded, reuse the engine.
        kwargs are passed to LLM check the vllm documentation for details.
        """
        global _VLLM_ENGINES
        self.model_path = model_path
        self.model_name = model_name or model_path
        
        if model_path in _VLLM_ENGINES:
            self.llm = _VLLM_ENGINES[model_path]
        else:
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError("vllm module not found. Please install vllm.")
                
            print(f"Loading VLLM model: {model_path}")
            # Default to some reasonable GPU util if not specified to allow multiple agents/models?
            # Actually user probably runs one model.
            # trust_remote_code=True is often needed.
            if "trust_remote_code" not in kwargs:
                kwargs["trust_remote_code"] = True
                
            self.llm = LLM(model=model_path, **kwargs)
            _VLLM_ENGINES[model_path] = self.llm
            
    def generate(self, prompt: str, sampling_params: Dict[str, Any]) -> str:
        from vllm import SamplingParams
        
        # Ensure we don't pass invalid params
        # Construct SamplingParams object
        sp = SamplingParams(**sampling_params)
        
        outputs = self.llm.generate([prompt], sp, use_tqdm=False)
        return outputs[0].outputs[0].text
