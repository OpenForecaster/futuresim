"""BasicAgent DataFrame query handling."""

from datetime import date
from typing import Dict, Any, Optional, Tuple

from agents.utils.df_interface import DfInterface


class QueryHandler:
    def __init__(self):
        self._df_interface: Optional[DfInterface] = None
    
    def setup(self, csv_path: str, forecast_interface, agent_id: str, current_date: date, single_agent_mode: bool = False) -> None:
        self._df_interface = DfInterface(csv_path, forecast_interface, agent_id, current_date, single_agent_mode)
    
    def execute(self, code: str, extra_context: dict = None) -> Tuple[str, Optional[str]]:
        if not self._df_interface:
            return "", "QueryHandler not initialized"
        return self._df_interface.execute_query(code, extra_context=extra_context)
    
    def get_info(self) -> Dict[str, Any]:
        if not self._df_interface:
            return {'n_rows': 0, 'n_active': 0, 'n_resolved': 0, 'columns': [], 'columns_desc': ''}
        return self._df_interface.get_info()
    
    def get_question_title(self, qid: str) -> Optional[str]:
        """Look up a question's title from the DataFrame."""
        if not self._df_interface:
            return None
        df = self._df_interface.load_df()
        match = df[df['qid'] == str(qid)]
        if not match.empty:
            return match.iloc[0].get('title')
        return None
    
    def invalidate_cache(self) -> None:
        if self._df_interface:
            self._df_interface.invalidate_cache()
