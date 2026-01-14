"""
Query execution for agent DataFrame interaction.

WARNING: Current implementation uses eval() which is NOT SAFE for untrusted code.
This is acceptable for testing with controlled agents, but needs proper sandboxing
for production use. Options to explore later:
- RestrictedPython
- AST whitelisting
- Subprocess isolation
- Docker/container execution

TODO: Add proper sandboxing before running untrusted agent code.
"""

import signal
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, Any


class TimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


class QueryExecutor:
    """
    Executes agent-provided pandas code.
    
    ⚠️ SAFETY WARNING: Uses eval() - not safe for untrusted input!
    Only use with controlled agent code during testing.
    """
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Code execution exceeded {self.timeout}s timeout")
    
    def execute(self, df: pd.DataFrame, code: str, 
                current_date: date = None,
                extra_context: dict = None) -> Tuple[str, Optional[str]]:
        """
        Execute agent code on the DataFrame.
        
        Args:
            df: The DataFrame to operate on (will use a copy)
            code: Python code to execute (should be an expression)
            current_date: The current simulation date (available as `today`)
            extra_context: Additional variables to make available
            
        Returns:
            (result_string, error_message)
            If successful: (formatted_result, None)
            If error: ("", error_message)
        """
        code = code.strip()
        if not code:
            return "", "Empty code provided"
        
        # Prepare execution environment
        df_copy = df.copy()
        exec_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'set': set,
                'tuple': tuple, 'range': range, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'True': True, 'False': False, 'None': None,
                'print': print,  # For debugging
            },
            'df': df_copy,
            'pd': pd,
            'date': date,
            'datetime': datetime,
            'timedelta': timedelta,
            'today': current_date or date.today(),
        }
        
        if extra_context:
            exec_globals.update(extra_context)
        
        # Set timeout (Unix only - on Windows this won't work)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(self.timeout))
        except (AttributeError, ValueError):
            # Windows or other platform without SIGALRM
            pass
        
        # Capture stdout to prevent agent prints from polluting terminal
        import io
        import sys
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = captured_output
            
            # Try as expression first
            try:
                result = eval(code, exec_globals, {})
            except SyntaxError:
                # If not an expression, try as statements
                exec(code, exec_globals, {})
                result = exec_globals.get('result', "Code executed (no result returned)")
            
            # Get any printed output
            printed = captured_output.getvalue()
            formatted_result = self._format_result(result)
            
            # Include printed output if any
            if printed.strip():
                formatted_result = f"{printed.strip()}\n\n{formatted_result}"
            
            return formatted_result, None
            
        except TimeoutError as e:
            return "", str(e)
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            # Reset alarm
            try:
                signal.alarm(0)
                if old_handler:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass
    
    def _format_result(self, result: Any) -> str:
        """Format the result for display to the agent."""
        if isinstance(result, pd.DataFrame):
            # Show shape info
            shape_info = f"DataFrame: {len(result)} rows × {len(result.columns)} columns\n"
            return shape_info + result.to_string()
        elif isinstance(result, pd.Series):
            return f"Series: {len(result)} items\n" + result.to_string()
        else:
            return str(result)
