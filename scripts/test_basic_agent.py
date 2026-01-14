#!/usr/bin/env python
"""
Test script for BasicAgent forecasting.

Usage (VLLM - local):
    python scripts/test_basic_agent.py --sim_name debug_run --start_date 2024-12-25 --end_date 2024-12-27

Usage (OpenRouter - API):
    export OPENROUTER_API_KEY="your-key-here"
    python scripts/test_basic_agent.py why --sim_name debug_run --start_date 2024-12-25 --end_date 2024-12-27

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
from inference.vllm import VLLMInference
from inference.openrouter import OpenRouterInference


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
    
    # Model settings
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
    
    # Save config
    save_config(output_dir, args)
    
    # Initialize inference provider
    inference_provider = None
    model_name = ""
    
    if not args.no_inference:
        if args.provider == "vllm":
            print(f"Loading VLLM model: {args.model_path} (max_model_len={args.max_model_len})")
            inference_provider = VLLMInference(args.model_path, max_model_len=args.max_model_len)
            model_name = os.path.basename(args.model_path)
            
        elif args.provider == "openrouter":
            print(f"Using OpenRouter model: {args.openrouter_model}")
            inference_provider = OpenRouterInference(args.openrouter_model)
            model_name = args.openrouter_model
    else:
        print("Running without LLM inference (--no_inference)")
    
    # Initialize environment with resolution date filter
    print(f"Loading dataset: {args.dataset}")
    env = SimulationEnvironment(
        dataset_name=args.dataset,
        start_date=sim_start,
        end_date=sim_end,
        context_dir=args.context_dir,
        inference_provider=inference_provider,
        output_dir=output_dir,
        resolution_start=resolution_start,
        resolution_end=resolution_end
    )
    
    # Create agent
    agent_config = AgentConfig(
        max_queries=args.max_queries,
        max_submit_retries=args.max_retries,
        memory_dir=output_dir,  # Persist memory in the simulation output directory
        sampling_params={
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
    )
    
    if inference_provider:
        # Derive agent prefix from model name (last part after '/', strip version/tags)
        agent_prefix = model_name.split('/')[-1].split(':')[0]
        
        # Get unique agent_id with numbering based on existing agents
        existing_prefixes = [a.agent_id.rsplit('_', 1)[0] for a in env.agents]
        count = sum(1 for p in existing_prefixes if p == agent_prefix)
        agent_id = f"{agent_prefix}_{count + 1:03d}"
        
        agent = BasicAgent(
            agent_id=agent_id,
            inference_provider=inference_provider,
            config=agent_config,
            model_name=model_name
        )
        env.add_agent(agent)
    else:
        print("No agent added (no inference provider)")
    
    # Run simulation
    print("\n" + "="*60)
    print("Starting simulation...")
    print("="*60 + "\n")
    
    env.run()
    
    print("\n" + "="*60)
    print(f"Simulation complete. Logs at: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
