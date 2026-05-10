"""
Query execution for agent DataFrame interaction.

WARNING: This still executes Python and is not a perfect sandbox.
It now validates queries to block file, process, network, and private-attribute
access that the agent does not need for DataFrame work, but proper isolation is
still preferable for fully untrusted code. Options to explore later:
- RestrictedPython
- AST whitelisting
- Subprocess isolation
- Docker/container execution

TODO: Add stronger isolation before running fully untrusted agent code.
"""

import ast
import builtins as py_builtins
import signal
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, Any


class TimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


class UnsafeQueryError(Exception):
    """Raised when a query tries to use blocked capabilities."""
    pass


_BLOCKED_NODE_TYPES = (
    ast.Global,
    ast.Nonlocal,
)

_BLOCKED_ATTR_NAMES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "builtins",
    "importlib",
    "ctypes",
    "requests",
    "urllib",
    "http",
    "pickle",
}

_BLOCKED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "help",
    "breakpoint",
}

_BLOCKED_CALL_ATTRS = {
    "eval",
    "tofile",
}

_BLOCKED_WRITER_METHODS = {
    "to_excel",
    "to_feather",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_gbq",
    "to_orc",
    "to_clipboard",
    "to_hdf",
    "to_stata",
}

_BLOCKED_PANDAS_CALL_PATHS = {
    "pd.ExcelWriter",
    "pd.HDFStore",
    "pandas.ExcelWriter",
    "pandas.HDFStore",
}

_BLOCKED_PATH_PREFIXES = (
    "pd.io",
    "pandas.io",
)

_ALLOWED_DUNDER_ATTRS = {
    "__name__",
}

_SAFE_IMPORT_MODULES = {
    "ast",
    "collections",
    "datetime",
    "itertools",
    "json",
    "math",
    "pandas",
    "random",
    "re",
    "statistics",
    "textwrap",
}

_SAFE_STRING_SERIALIZER_METHODS = {
    "to_csv",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_xml",
}

_BUFFER_KWARGS = {
    "buf",
    "filepath_or_buffer",
    "path",
    "path_or_buf",
    "path_or_buffer",
}

_PANDAS_IMPORT_ALIASES = {
    None,
    "pd",
}


def _full_attr_path(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _is_safe_import_target(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in _SAFE_IMPORT_MODULES


def _is_safe_import_alias(name: str, asname: str | None) -> bool:
    root = name.split(".", 1)[0]
    if root != "pandas":
        return True
    return asname in _PANDAS_IMPORT_ALIASES


def _safe_import(
    name: str,
    globals_dict: dict | None = None,
    locals_dict: dict | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    if level != 0 or not _is_safe_import_target(name):
        raise ImportError(f"Import not allowed in query_df: {name}")
    return py_builtins.__import__(name, globals_dict, locals_dict, fromlist, level)


def _extract_string_arg(node: ast.Call, keyword_name: str) -> Optional[str]:
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
        return None
    for kw in node.keywords:
        if kw.arg == keyword_name:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
            return None
    return None


def _is_safe_serializer_call(node: ast.Call) -> bool:
    if node.args:
        return False
    for kw in node.keywords:
        if kw.arg in _BUFFER_KWARGS:
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return False
    return True


def _is_safe_query_expr(expr: str) -> bool:
    return "@" not in expr and "__" not in expr


def _is_safe_df_query_call(node: ast.Call) -> bool:
    if len(node.args) > 1:
        return False
    for kw in node.keywords:
        if kw.arg in {"global_dict", "local_dict", "resolvers"}:
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return False
    expr = _extract_string_arg(node, "expr")
    if expr is None:
        return False
    return _is_safe_query_expr(expr)


class _QuerySafetyVisitor(ast.NodeVisitor):
    """Reject operations outside read/manipulate-in-memory DataFrame usage."""

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _BLOCKED_NODE_TYPES):
            raise UnsafeQueryError(
                f"{type(node).__name__} statements are not allowed in query_df"
            )
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") and node.attr not in _ALLOWED_DUNDER_ATTRS:
            raise UnsafeQueryError(
                f"Private attribute access is not allowed in query_df: {node.attr}"
            )
        if node.attr in _BLOCKED_ATTR_NAMES:
            raise UnsafeQueryError(
                f"Blocked attribute access in query_df: {node.attr}"
            )

        path = _full_attr_path(node)
        if path:
            for prefix in _BLOCKED_PATH_PREFIXES:
                if path == prefix or path.startswith(prefix + "."):
                    raise UnsafeQueryError(
                        f"Blocked module access in query_df: {path}"
                    )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _is_safe_import_target(alias.name) or not _is_safe_import_alias(alias.name, alias.asname):
                raise UnsafeQueryError(
                    f"Import not allowed in query_df: {alias.name}"
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if (
            node.level != 0
            or not module
            or not _is_safe_import_target(module)
            or module == "pandas"
        ):
            raise UnsafeQueryError(
                f"Import not allowed in query_df: {module or '<relative import>'}"
            )

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _BLOCKED_CALL_NAMES:
            raise UnsafeQueryError(
                f"Blocked function call in query_df: {func.id}"
            )

        if isinstance(func, ast.Attribute):
            path = _full_attr_path(func)
            if func.attr == "query":
                if not _is_safe_df_query_call(node):
                    raise UnsafeQueryError(
                        "Blocked df.query call in query_df: only literal query strings "
                        "without @ references or custom eval context are allowed"
                    )
            elif func.attr in _SAFE_STRING_SERIALIZER_METHODS:
                if not _is_safe_serializer_call(node):
                    raise UnsafeQueryError(
                        f"Blocked output method in query_df: {func.attr}"
                    )
            if func.attr in _BLOCKED_CALL_ATTRS:
                raise UnsafeQueryError(
                    f"Blocked method call in query_df: {func.attr}"
                )
            if func.attr in _BLOCKED_WRITER_METHODS:
                raise UnsafeQueryError(
                    f"Blocked output method in query_df: {func.attr}"
                )
            if path in _BLOCKED_PANDAS_CALL_PATHS:
                raise UnsafeQueryError(
                    f"Blocked pandas constructor in query_df: {path}"
                )
            if path and (
                path.startswith("pd.read_") or path.startswith("pandas.read_")
            ):
                raise UnsafeQueryError(
                    f"Blocked pandas reader in query_df: {path}"
                )

        self.generic_visit(node)


def validate_query_code(code: str) -> None:
    """Parse and validate agent query code before execution."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        raise
    _QuerySafetyVisitor().visit(tree)


class QueryExecutor:
    """
    Executes agent-provided pandas code.
    
    Uses restricted globals plus AST validation. This is still not a perfect
    sandbox, but it blocks the obvious file/process/network escape routes that
    are unnecessary for DataFrame inspection.
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

        try:
            validate_query_code(code)
        except SyntaxError as e:
            return "", f"SyntaxError: {e}"
        except UnsafeQueryError as e:
            return "", f"UnsafeQueryError: {e}"
        
        # Prepare execution environment
        df_copy = df.copy()
        exec_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float, 'type': type,
                'bool': bool, 'list': list, 'dict': dict, 'set': set,
                'tuple': tuple, 'range': range, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'True': True, 'False': False, 'None': None,
                'print': print,  # For debugging
                'dir': dir,
                '__import__': _safe_import,
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
