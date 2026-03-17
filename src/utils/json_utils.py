import ast
import json
import logging

logger = logging.getLogger(__name__)

def safe_parse_python_literal(text: str, max_depth: int = 100, max_length: int = 100000) -> any:
    """
    Safely parse a string that might be a Python literal (e.g., from an LLM) or JSON.

    This function provides a safer alternative to ast.literal_eval by:
    1. Limiting the total length of the input string.
    2. Limiting the nesting depth of structures (dicts, lists, tuples).
    3. Attempting standard JSON parsing first.

    Args:
        text: The string to parse.
        max_depth: Maximum allowed nesting depth of brackets and braces.
        max_length: Maximum allowed length of the input string.

    Returns:
        The parsed data (typically a dict or list).

    Raises:
        ValueError: If the input is too long, too deep, or cannot be parsed.
    """
    if not text:
        return None

    if not isinstance(text, str):
        return text

    if len(text) > max_length:
        logger.warning(f"Rejected parsing of string: length {len(text)} exceeds limit of {max_length}")
        raise ValueError(f"Input length exceeds limit of {max_length}")

    # 1. Check nesting depth to prevent Denial of Service via deeply nested structures
    if not _check_nesting_depth(text, max_depth):
        logger.warning(f"Rejected parsing of string: nesting depth exceeds limit of {max_depth}")
        raise ValueError(f"Nesting depth exceeds limit of {max_depth}")

    # 2. Try standard JSON first (it's the most robust and preferred)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Use ast.literal_eval as a fallback for Python-style literals
    # In modern Python (3.9+), ast.literal_eval is relatively safe but still vulnerable to DoS.
    # Our depth check above mitigates the DoS risk.
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError, MemoryError) as e:
        logger.debug(f"Failed to parse literal: {e}")
        raise ValueError(f"Failed to parse literal: {e}")

def _check_nesting_depth(text: str, max_depth: int) -> bool:
    """Check if the nesting depth of [], {}, and () exceeds max_depth."""
    depth = 0
    in_string = False
    quote_char = None
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char in '"\'':
            if not in_string:
                in_string = True
                quote_char = char
            elif char == quote_char:
                in_string = False
                quote_char = None
            continue

        if not in_string:
            if char in '[{(':
                depth += 1
                if depth > max_depth:
                    return False
            elif char in ']})':
                depth -= 1

    # We don't strictly require depth == 0 here as ast.literal_eval
    # will handle unbalanced structures by raising a SyntaxError.
    return True
