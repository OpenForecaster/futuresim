#!/usr/bin/env python
"""
Test script for BasicAgent forecasting.

Usage (Single agent - VLLM):
    python scripts/test_basic_agent.py --sim_name debug_run --start_date 2024-12-25 --end_date 2024-12-27

Usage (Single agent - OpenRouter):
    export OPENROUTER_API_KEY="your-key-here"
    python scripts/test_basic_agent.py --provider openrouter --openrouter_model xiaomi/mimo-v2-flash:free --sim_name debug_run --start_date 2024-12-25 --end_date 2024-12-27

Usage (Multi-agent - config file):
    python scripts/test_basic_agent.py --agents_config configs/agents_example.yaml --sim_name multi_agent_run --start_date 2024-12-25 --end_date 2024-12-27

For interactive GPU session on cluster (VLLM only):
    condor_submit_bid 25 -i -append request_gpus=1 -append "requirements=TARGET.CUDACapability == 8.0" -append request_memory=40960
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env file if exists
env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), '.env')
if os.path.exists(env_path):
    print(f"Loading environment from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

from environment.env import SimulationEnvironment
from agents.basicAgent import BasicAgent, AgentConfig


# Default paths
DATASET_PATH = "/is/cluster/fast/sgoel/forecasting/qs/OpenForesight/data/"
MODEL_PATH = "/is/cluster/fast/rolmedo/models/qwen3-4b-it-2507"
CURRENT_SIM_DIR = "/is/cluster/fast/sgoel/forecasting/current_sim"


def create_output_dir(sim_name: str, base_dir: str = CURRENT_SIM_DIR) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(base_dir, sim_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(output_dir: str, args: argparse.Namespace, extra: dict = None):
    """Save run configuration to config.json."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    if extra:
        config.update(extra)
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved to {config_path}")


def load_agents_config(config_path: str) -> dict:
    """Load agents configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config files. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_inference_provider(provider: str, model: str, args):
    """Create an inference provider instance."""
    if provider == "vllm":
        from inference.vllm import VLLMInference
        return VLLMInference(model, max_model_len=args.max_model_len)
    elif provider == "openrouter":
        from inference.openrouter import OpenRouterInference
        return OpenRouterInference(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_model_short_name(model: str) -> str:
    """Extract short model name from full model path/ID."""
    # Handle paths like /path/to/model-name
    name = os.path.basename(model)
    # Handle OpenRouter IDs like vendor/model-name:variant
    if '/' in model:
        name = model.split('/')[-1]
    # Remove version tags
    name = name.split(':')[0]
    return name


def create_agents_from_config(config: dict, args, output_dir: str) -> list:
    """Create agent instances from config dict."""
    from inference.openrouter import GlobalRateLimiter
    
    defaults = config.get('defaults', {})
    agents_list = config.get('agents', [])
    
    if not agents_list:
        raise ValueError("No agents defined in config file")
    
    print(f"  Found {len(agents_list)} agents in config", flush=True)
    
    # Configure global rate limiter
    GlobalRateLimiter.configure(args.rate_limit)
    print(f"  Rate limiter configured: {args.rate_limit} req/s", flush=True)
    
    agents = []
    agent_counts = {}  # Track counts per scaffold+model combo for unique IDs
    
    for i, agent_def in enumerate(agents_list):
        # Merge defaults with agent-specific settings
        provider = agent_def.get('provider', defaults.get('provider', 'openrouter'))
        scaffold = agent_def.get('scaffold', defaults.get('scaffold', 'basic'))
        model = agent_def.get('model')
        
        if not model:
            raise ValueError("Each agent must specify a 'model'")
        
        print(f"  [{i+1}/{len(agents_list)}] Creating {scaffold}:{provider}:{model}...", flush=True)
        
        # Create inference provider
        inference_provider = create_inference_provider(provider, model, args)
        
        # Build agent config with merged settings
        max_queries = agent_def.get('max_queries', defaults.get('max_queries', args.max_queries))
        max_retries = agent_def.get('max_retries', defaults.get('max_retries', args.max_retries))
        temperature = agent_def.get('temperature', defaults.get('temperature', args.temperature))
        max_tokens = agent_def.get('max_tokens', defaults.get('max_tokens', args.max_tokens))
        
        # Generate unique agent ID: scaffold_modelname_NNN
        model_short = get_model_short_name(model)
        base_id = f"{scaffold}_{model_short}"
        count = agent_counts.get(base_id, 0) + 1
        agent_counts[base_id] = count
        agent_id = f"{base_id}_{count:03d}"
        
        # Create agent directory for logging
        agent_dir = os.path.join(output_dir, "agents", agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        agent_config = AgentConfig(
            max_queries=max_queries,
            max_submit_retries=max_retries,
            memory_dir=agent_dir,  # Per-agent memory directory
            sampling_params={
                'temperature': temperature,
                'max_tokens': max_tokens,
            }
        )
        
        # Currently only support BasicAgent scaffold
        if scaffold != 'basic':
            raise ValueError(f"Unknown scaffold: {scaffold}. Only 'basic' is supported.")
        
        agent = BasicAgent(
            agent_id=agent_id,
            inference_provider=inference_provider,
            config=agent_config,
            model_name=model
        )
        agents.append(agent)
        print(f"  Created agent: {agent_id} ({provider}:{model})")
    
    return agents


def main():
    parser = argparse.ArgumentParser(description="Test BasicAgent Forecasting")
    
    # Simulation settings
    parser.add_argument("--sim_name", default="debug_sim",
                       help="Name for this simulation run")
    parser.add_argument("--start_date", default="2024-12-25",
                       help="First resolution date (YYYY-MM-DD) - questions resolving from this date")
    parser.add_argument("--end_date", default="2024-12-27",
                       help="Last resolution date (YYYY-MM-DD) - questions resolving until this date")
    parser.add_argument("--lookback_days", type=int, default=7,
                       help="Days before first resolution to start simulation (default 7)")
    
    # Data paths
    parser.add_argument("--dataset", default=DATASET_PATH,
                       help="Path to OpenForesight dataset")
    parser.add_argument("--context_dir", default="",
                       help="Path to context/news directory (optional)")
    parser.add_argument("--output_base", default=CURRENT_SIM_DIR,
                       help="Base directory for simulation outputs")
    
    # Multi-agent config
    parser.add_argument("--agents_config", default=None,
                       help="Path to YAML config file for multi-agent runs")
    parser.add_argument("--rate_limit", type=float, default=32.0,
                       help="OpenRouter rate limit (requests/second, default 32)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run agents in parallel (default: True)")
    parser.add_argument("--no_parallel", action="store_true",
                       help="Disable parallel agent execution")
    
    # Single-agent settings (used when no config file)
    parser.add_argument("--provider", choices=["vllm", "openrouter"], default="vllm",
                       help="Inference provider: 'vllm' (local) or 'openrouter' (API)")
    parser.add_argument("--model_path", default=MODEL_PATH,
                       help="Path to model for VLLM inference")
    parser.add_argument("--openrouter_model", default="xiaomi/mimo-v2-flash:free",
                       help="Model ID for OpenRouter (e.g., 'xiaomi/mimo-v2-flash:free')")
    parser.add_argument("--max_model_len", type=int, default=32768,
                       help="Max context length for VLLM (default 32768)")
    parser.add_argument("--no_inference", action="store_true",
                       help="Run without LLM (for testing setup)")
    
    # Agent settings
    parser.add_argument("--max_queries", type=int, default=3,
                       help="Max DataFrame queries per day")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Max retry attempts for forecast parsing")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max tokens to generate")
    
    args = parser.parse_args()
    
    # Handle parallel flag
    if args.no_parallel:
        args.parallel = False
    
    # Parse dates - these are resolution date bounds
    resolution_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    resolution_end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    # Simulation starts lookback_days before first resolution
    from datetime import timedelta
    sim_start = resolution_start - timedelta(days=args.lookback_days)
    sim_end = resolution_end
    
    print(f"Resolution window: {resolution_start} to {resolution_end}")
    print(f"Simulation window: {sim_start} to {sim_end} (lookback={args.lookback_days} days)")
    
    # Create output directory
    output_dir = create_output_dir(args.sim_name, args.output_base)
    print(f"Output directory: {output_dir}")
    
    # Create agents directory
    agents_dir = os.path.join(output_dir, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    
    # Save config
    save_config(output_dir, args)
    
    # Determine agents to create
    agents = []
    
    if args.agents_config:
        # Multi-agent mode from config file
        print(f"\nLoading agents from config: {args.agents_config}", flush=True)
        config = load_agents_config(args.agents_config)
        print(f"  Config loaded successfully", flush=True)
        agents = create_agents_from_config(config, args, output_dir)
        
        # Save agent config copy
        import shutil
        config_copy = os.path.join(output_dir, "agents_config.yaml")
        shutil.copy(args.agents_config, config_copy)
        
    elif not args.no_inference:
        # Single agent mode (legacy)
        from inference.openrouter import GlobalRateLimiter
        GlobalRateLimiter.configure(args.rate_limit)
        
        if args.provider == "vllm":
            from inference.vllm import VLLMInference
            print(f"Loading VLLM model: {args.model_path} (max_model_len={args.max_model_len})")
            inference_provider = VLLMInference(args.model_path, max_model_len=args.max_model_len)
            model_name = os.path.basename(args.model_path)
            
        elif args.provider == "openrouter":
            from inference.openrouter import OpenRouterInference
            print(f"Using OpenRouter model: {args.openrouter_model}")
            inference_provider = OpenRouterInference(args.openrouter_model)
            model_name = args.openrouter_model
        
        # Create agent
        model_short = get_model_short_name(model_name)
        agent_id = f"basic_{model_short}_001"
        
        # Create agent directory
        agent_dir = os.path.join(output_dir, "agents", agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        
        agent_config = AgentConfig(
            max_queries=args.max_queries,
            max_submit_retries=args.max_retries,
            memory_dir=agent_dir,
            sampling_params={
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
            }
        )
        
        agent = BasicAgent(
            agent_id=agent_id,
            inference_provider=inference_provider,
            config=agent_config,
            model_name=model_name
        )
        agents.append(agent)
        print(f"  Created agent: {agent_id}")
    else:
        print("Running without LLM inference (--no_inference)")
    
    # Initialize environment with resolution date filter
    print(f"\nLoading dataset: {args.dataset}")
    env = SimulationEnvironment(
        dataset_name=args.dataset,
        start_date=sim_start,
        end_date=sim_end,
        context_dir=args.context_dir,
        inference_provider=None,  # Not used with multi-agent
        output_dir=output_dir,
        resolution_start=resolution_start,
        resolution_end=resolution_end,
        parallel=args.parallel,
    )
    
    # Add all agents
    for agent in agents:
        env.add_agent(agent)
    
    if not agents:
        print("No agents added")
        return
    
    # Run simulation
    print("\n" + "="*60)
    print(f"Starting simulation with {len(agents)} agent(s)...")
    if args.parallel:
        print(f"Parallel execution: ENABLED")
    print("="*60 + "\n")
    
    env.run()
    
    print("\n" + "="*60)
    print(f"Simulation complete. Logs at: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
