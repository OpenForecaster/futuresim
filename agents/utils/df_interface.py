"""
DfInterface: DataFrame interface for agent question exploration.

Loads market.csv and provides query execution capabilities.
This is an agent-side utility - the environment doesn't need to know how agents query data.
"""

import json
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from environment.safe_executor import QueryExecutor


class DfInterface:
    """
    DataFrame interface for exploring forecasting questions.
    
    Loads market.csv and enriches it with agent-specific columns
    (my_prediction, my_prediction_date) from the forecast interface.
    
    Provides query execution via sandboxed eval().
    """
    
    def __init__(self, 
                 csv_path: str,
                 forecast_interface,  # Has get_agent_predictions(agent_id)
                 agent_id: str,
                 current_date: date,
                 single_agent_mode: bool = False):
        """
        Initialize the DataFrame interface.
        
        Args:
            csv_path: Path to market.csv
            forecast_interface: Interface to get agent's predictions
            agent_id: Current agent's ID
            current_date: Current simulation date
            single_agent_mode: If True, hide market_aggregate (not meaningful for single agent)
        """
        self.csv_path = Path(csv_path)
        self.forecast_interface = forecast_interface
        self.agent_id = agent_id
        self.current_date = current_date
        self.single_agent_mode = single_agent_mode
        
        self._df: Optional[pd.DataFrame] = None
        self._executor = QueryExecutor(timeout_seconds=5.0)
    
    def load_df(self) -> pd.DataFrame:
        """
        Load and cache the DataFrame with agent-specific columns.
        
        Returns:
            DataFrame with all question data plus my_prediction columns
        """
        if self._df is not None:
            return self._df
        
        # Load base CSV. Some simulation days can legitimately have no active
        # questions, producing an empty market.csv file.
        try:
            df = pd.read_csv(self.csv_path, dtype={'qid': str})
        except EmptyDataError:
            df = pd.DataFrame(columns=[
                "qid",
                "title",
                "answer_type",
                "is_resolved",
                "close_time",
                "resolution_date",
                "resolution",
                "market_aggregate",
                "options",
                "num_predictions",
            ])
        
        # Parse JSON columns
        if 'market_aggregate' in df.columns:
            df['market_aggregate'] = df['market_aggregate'].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else None
            )
        
        if 'options' in df.columns:
            df['options'] = df['options'].apply(
                lambda x: json.loads(x) if pd.notna(x) and x else None
            )
        
        # Parse dates
        if 'resolution_date' in df.columns:
            df['resolution_date'] = pd.to_datetime(df['resolution_date']).dt.date
        
        # Add agent-specific columns
        agent_preds = self.forecast_interface.get_agent_predictions(self.agent_id)
        
        df['my_prediction'] = df['qid'].apply(
            lambda qid: agent_preds.get(qid, {}).get('outcomes')
        )
        df['my_prediction_date'] = df['qid'].apply(
            lambda qid: agent_preds.get(qid, {}).get('date')
        )
        
        # In single-agent mode, drop market_aggregate (it just reflects own predictions)
        if self.single_agent_mode and 'market_aggregate' in df.columns:
            df = df.drop(columns=['market_aggregate'])
        
        self._df = df
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get DataFrame schema info for agent prompting.
        
        Returns:
            Dict with n_rows, n_active, n_resolved, columns, columns_desc
        """
        df = self.load_df()
        
        columns_desc = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if col == 'qid':
                dtype = 'str'  # Explicitly tell agent it's a string
            columns_desc.append(f"- {col} ({dtype})")
        
        n_active = len(df[df['is_resolved'] == False]) if 'is_resolved' in df.columns else len(df)
        n_resolved = len(df[df['is_resolved'] == True]) if 'is_resolved' in df.columns else 0
        
        return {
            'n_rows': len(df),
            'n_active': n_active,
            'n_resolved': n_resolved,
            'columns': list(df.columns),
            'columns_desc': "\n".join(columns_desc),
        }
    
    def execute_query(self, code: str, extra_context: dict = None) -> Tuple[str, Optional[str]]:
        """
        Execute pandas code on the DataFrame.

        Args:
            code: Python code to execute (should be an expression or statements)
            extra_context: Additional variables to inject into the sandbox

        Returns:
            (result_string, error_message)
            If successful: (formatted_result, None)
            If error: ("", error_message)
        """
        df = self.load_df()
        return self._executor.execute(
            df, code,
            current_date=self.current_date,
            extra_context=extra_context
        )
    
    def invalidate_cache(self) -> None:
        """Clear the cached DataFrame (e.g., after submitting a prediction)."""
        self._df = None
