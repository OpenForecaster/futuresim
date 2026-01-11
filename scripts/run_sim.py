import argparse
import sys
import os
import random
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.env import SimulationEnvironment
from environment.interfaces import PredictionSubmission
from agents.base import BaseAgent

try:
    from inference.vllm import VLLMInference
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class RandomProbAgent(BaseAgent):
    """Stub agent that predicts random probabilities."""
    
    def act(self, doc_interface, forecast_interface, current_date):
        qs = forecast_interface.list_questions()
        if not qs:
            return []
        
        actions = []
        for q in qs[:3]:
            outcomes = {}
            remaining = 1.0
            n_outcomes = random.randint(1, 3)
            outcome_names = ["Yes", "No", "Maybe"][:n_outcomes]
            
            for i, name in enumerate(outcome_names):
                if i == len(outcome_names) - 1:
                    prob = remaining * random.random()
                else:
                    prob = remaining * random.random() * 0.7
                outcomes[name] = round(prob, 3)
                remaining -= prob
            
            pred = PredictionSubmission(question_id=q.id, outcomes=outcomes)
            forecast_interface.submit_prediction(pred)
            print(f"  [{self.agent_id}] Q: {q.title[:30]}... → {outcomes}")
            actions.append({"q": q.id, "outcomes": outcomes})
            
        return actions


class InformedStubAgent(BaseAgent):
    """Stub agent that uses current aggregate + noise."""
    
    def act(self, doc_interface, forecast_interface, current_date):
        qs = forecast_interface.list_questions()
        if not qs:
            return []
        
        actions = []
        for q in qs[:3]:
            outcomes = {}
            
            if q.aggregate:
                for outcome, prob in q.aggregate.items():
                    noisy_prob = prob + random.gauss(0, 0.1)
                    outcomes[outcome] = max(0.01, min(0.99, noisy_prob))
            else:
                outcomes["Yes"] = random.uniform(0.3, 0.7)
                outcomes["No"] = 1.0 - outcomes["Yes"] - 0.1
            
            total = sum(outcomes.values())
            if total > 1:
                outcomes = {k: v/total for k, v in outcomes.items()}
            outcomes = {k: round(v, 3) for k, v in outcomes.items()}
            
            pred = PredictionSubmission(question_id=q.id, outcomes=outcomes)
            forecast_interface.submit_prediction(pred)
            print(f"  [{self.agent_id}] Q: {q.title[:30]}... → {outcomes}")
            actions.append({"q": q.id, "outcomes": outcomes})
            
        return actions


def main():
    parser = argparse.ArgumentParser(description="Run Forecasting Simulation")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset path")
    parser.add_argument("--context_dir", default="data/context")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_path", help="Path to model for VLLM")
    parser.add_argument("--inference_provider", choices=["vllm", "none"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    inference_provider = None
    if args.inference_provider == "vllm":
        if not VLLM_AVAILABLE:
            print("Error: vllm not installed")
            sys.exit(1)
        if not args.model_path:
            print("Error: --model_path required for vllm")
            sys.exit(1)
        inference_provider = VLLMInference(args.model_path)
            
    print("Initializing Environment...")
    env = SimulationEnvironment(
        dataset_name=args.dataset,
        start_date=start,
        end_date=end,
        context_dir=args.context_dir,
        inference_provider=inference_provider,
        output_dir=args.output_dir
    )
    
    agent1 = RandomProbAgent(agent_id="random_agent")
    agent2 = InformedStubAgent(agent_id="informed_agent")
    env.add_agent(agent1)
    env.add_agent(agent2)
    
    print("Running Simulation...")
    env.run()


if __name__ == "__main__":
    main()
