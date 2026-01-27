"""
FeedbackHandler: Computes and formatting daily feedback for the agent.
"""

from typing import Dict, Any, List, Set, Optional
from datetime import date
from collections import defaultdict

from environment.scoring import (
    BrierScorer, 
    resolve_question, 
    compute_snapshot_peer_scores,
    PredictionHistory,
    DEFAULT_SCORER
)
from environment.ansmatching import AnswerMatcher
from environment.data_loader import Question

class FeedbackHandler:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Cumulative metrics tracking
        self.total_brier_sum = 0.0
        self.total_tw_peer_sum = 0.0
        self.total_accuracy_count = 0
        self.total_resolved_count = 0
        
        # Track which questions we've already processed to avoid double-counting
        self.processed_qids: Set[str] = set()
        
        # Matcher instance (lazy initialized)
        self._matcher: Optional[AnswerMatcher] = None
        
    def _get_matcher(self, inference_provider):
        if self._matcher is None:
            # Create a matcher using the agent's inference provider
            # We use a mocked logger or None since we don't need to log matching details here
            self._matcher = AnswerMatcher(inference_provider, logger=None)
        return self._matcher
        
    def generate_feedback(self, 
                          forecast_interface, 
                          current_date: date,
                          inference_provider) -> Dict[str, Any]:
        """
        Generate feedback for the day including:
        1. Results for questions resolved since last check
        2. Cumulative performance metrics
        """
        matcher = self._get_matcher(inference_provider)
        
        # 1. Identify newly resolved questions
        resolved_today = []
        
        # Check all resolved key in the interface
        # Note: forecast_interface.resolved_questions contains ALL resolved questions
        for q in forecast_interface.resolved_questions:
            if q.qid in self.processed_qids:
                continue
                
            # It's a new resolved question! Process it.
            self.processed_qids.add(q.qid)
            
            # Get prediction history
            history = forecast_interface.histories.get(q.qid)
            if not history:
                continue
                
            # Check if WE predicted on it
            # Using get_prediction_as_of with resolution date to get our final standing
            my_final_pred = history.get_prediction_as_of(self.agent_id, q.resolution_date)
            
            if not my_final_pred:
                # We didn't participate, so no score impact (for now)
                # (Though technically abstaining = 0 peer score, but metrics usually track active participation)
                continue
                
            # --- Compute Scores ---
            
            # 1. Brier Score
            brier = DEFAULT_SCORER.score_prediction(
                my_final_pred, 
                q.ground_truth_answer, 
                matcher,
                question_id=q.qid,
                question_title=q.title
            )
            
            # 2. Accuracy (Top choice match)
            # Find outcome with max probability
            best_outcome = max(my_final_pred.outcomes.items(), key=lambda x: x[1])[0] if my_final_pred.outcomes else None
            is_accurate = False
            if best_outcome:
                if matcher.is_equivalent(best_outcome, q.ground_truth_answer,
                                       question_id=q.qid, question_title=q.title):
                    is_accurate = True
            
            # 3. Time-Weighted Peer Score
            # We need to resolve the whole question to get the peer scores
            q_result = resolve_question(
                history, 
                q.ground_truth_answer, 
                matcher, 
                DEFAULT_SCORER
            )
            tw_peer_score = q_result.agent_scores.get(self.agent_id, 0.0)
            
            # --- Update Cumulative Metrics ---
            self.total_brier_sum += brier
            self.total_tw_peer_sum += tw_peer_score
            if is_accurate:
                self.total_accuracy_count += 1
            self.total_resolved_count += 1
            
            # --- Record for "Today's Results" ---
            resolved_today.append({
                'qid': q.qid,
                'title': q.title,
                'my_pred_outcome': best_outcome,
                'my_pred_prob': my_final_pred.outcomes.get(best_outcome, 0.0) if best_outcome else 0.0,
                'ground_truth': q.ground_truth_answer,
                'brier': brier,
                'tw_peer': tw_peer_score
            })
            
        # 2. Compute Total Stats
        # Total predictions = Resolved (participated) + Active (participated)
        
        # Count active participations
        active_count = 0
        all_histories = forecast_interface.histories
        for qid, history in all_histories.items():
            # If not in processed_qids (meaning not resolved and processed), check if we have a prediction
            if qid not in self.processed_qids and history.get_latest_prediction(self.agent_id):
                active_count += 1
                
        total_predictions = self.total_resolved_count + active_count
        
        avg_brier = (self.total_brier_sum / self.total_resolved_count) if self.total_resolved_count > 0 else 0.0
        accuracy = (self.total_accuracy_count / self.total_resolved_count * 100) if self.total_resolved_count > 0 else 0.0
        
        return {
            'resolved_today': resolved_today,
            'metrics': {
                'total_predictions': total_predictions,
                'num_resolved': self.total_resolved_count,
                'accuracy': accuracy,
                'avg_brier': avg_brier,
                'tw_peer_score': self.total_tw_peer_sum
            }
        }

    def format_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Convert feedback dict to string for the prompt."""
        sections = []
        
        # Section 1: Today's Resolved Questions
        resolved = feedback_data.get('resolved_today', [])
        if resolved:
            lines = ["## YESTERDAY'S RESULTS", ""]
            for item in resolved:
                lines.append(f"- \"{item['title']}\"")
                lines.append(f"  Your prediction: {item['my_pred_outcome']} ({item['my_pred_prob']:.2f}) | Truth: {item['ground_truth']}")
                lines.append(f"  Brier: {item['brier']:+.2f} | TW-Peer: {item['tw_peer']:+.2f}")
                lines.append("") 
            sections.append("\n".join(lines))
        
        # Section 2: Cumulative Performance
        # Only show if we have at least one resolved question or prediction
        metrics = feedback_data.get('metrics', {})
        if metrics.get('total_predictions', 0) > 0:
            m_lines = ["## YOUR CUMULATIVE PERFORMANCE"]
            m_lines.append(f"- Total Predictions: {metrics['total_predictions']} ({metrics['num_resolved']} resolved)")
            
            if metrics['num_resolved'] > 0:
                m_lines.append(
                    f"- accuracy: {metrics['accuracy']:.1f}% | "
                    f"avg brier: {metrics['avg_brier']:.3f} | "
                    f"time weighted peer: {metrics['tw_peer_score']:.2f}"
                )
            else:
                m_lines.append("- (Waiting for resolutions to compute scores)")
            
            sections.append("\n".join(m_lines))
            
        return "\n\n".join(sections)
