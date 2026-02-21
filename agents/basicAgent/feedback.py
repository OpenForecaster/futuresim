"""
FeedbackHandler: Computes and formatting daily feedback for the agent.
"""

from typing import Dict, Any, Set, Optional, Callable
from datetime import date

class FeedbackHandler:
    def __init__(self, agent_id: str, timing_callback: Optional[Callable[[float], None]] = None):
        self.agent_id = agent_id
        # Kept for backward compatibility with constructor call sites.
        # Feedback no longer runs matcher calls locally.
        self._timing_callback = timing_callback
        
        # Cumulative metrics tracking
        self.total_brier_sum = 0.0
        self.total_tw_peer_sum = 0.0
        self.total_accuracy_count = 0
        self.total_resolved_count = 0
        
        # Track which questions we've already processed to avoid double-counting
        self.processed_qids: Set[str] = set()
        
    def generate_feedback(self, 
                          forecast_interface, 
                          current_date: date,
                          inference_provider) -> Dict[str, Any]:
        """
        Generate feedback for the day including:
        1. Results for questions resolved since last check
        2. Cumulative performance metrics
        """
        del current_date
        del inference_provider
        
        # 1. Identify newly resolved questions
        resolved_today = []

        # Consume authoritative env-produced resolution summaries.
        for event in getattr(forecast_interface, "resolution_events", []):
            qid = str(event.get("qid")) if event.get("qid") is not None else None
            if not qid or qid in self.processed_qids:
                continue
            self.processed_qids.add(qid)

            per_agent = event.get("agents", {}) or {}
            my_stats = per_agent.get(self.agent_id)
            if not isinstance(my_stats, dict):
                continue

            brier_raw = my_stats.get("brier")
            if brier_raw is None:
                continue

            brier = float(brier_raw)
            tw_peer_score = float(my_stats.get("tw_peer", 0.0))
            is_accurate = bool(my_stats.get("is_accurate", False))
            best_outcome = my_stats.get("best_outcome")
            best_prob = float(my_stats.get("best_prob", 0.0) or 0.0)

            self.total_brier_sum += brier
            self.total_tw_peer_sum += tw_peer_score
            if is_accurate:
                self.total_accuracy_count += 1
            self.total_resolved_count += 1

            resolved_today.append({
                'qid': qid,
                'title': event.get("title", ""),
                'my_pred_outcome': best_outcome,
                'my_pred_prob': best_prob,
                'ground_truth': event.get("ground_truth", ""),
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

    def format_feedback(self, feedback_data: Dict[str, Any], show_tw_peer: bool = True) -> str:
        """Convert feedback dict to string for the prompt."""
        sections = []
        
        # Section 1: Today's Resolved Questions
        resolved = feedback_data.get('resolved_today', [])
        if resolved:
            lines = ["## YESTERDAY'S RESULTS", ""]
            for item in resolved:
                lines.append(f"- \"{item['title']}\"")
                lines.append(f"  Your prediction: {item['my_pred_outcome']} ({item['my_pred_prob']:.2f}) | Truth: {item['ground_truth']}")
                if show_tw_peer:
                    lines.append(f"  Brier: {item['brier']:+.2f} | TW-Peer: {item['tw_peer']:+.2f}")
                else:
                    lines.append(f"  Brier: {item['brier']:+.2f}")
                lines.append("") 
            sections.append("\n".join(lines))
        
        # Section 2: Cumulative Performance
        # Only show if we have at least one resolved question or prediction
        metrics = feedback_data.get('metrics', {})
        if metrics.get('total_predictions', 0) > 0:
            m_lines = ["## YOUR CUMULATIVE PERFORMANCE"]
            m_lines.append(f"- Total Predictions: {metrics['total_predictions']} ({metrics['num_resolved']} resolved)")
            
            if metrics['num_resolved'] > 0:
                if show_tw_peer:
                    m_lines.append(
                        f"- accuracy: {metrics['accuracy']:.1f}% | "
                        f"avg brier: {metrics['avg_brier']:.3f} | "
                        f"time weighted peer: {metrics['tw_peer_score']:.2f}"
                    )
                else:
                    m_lines.append(
                        f"- accuracy: {metrics['accuracy']:.1f}% | "
                        f"avg brier: {metrics['avg_brier']:.3f}"
                    )
            else:
                m_lines.append("- (Waiting for resolutions to compute scores)")
            
            sections.append("\n".join(m_lines))
            
        return "\n\n".join(sections)
