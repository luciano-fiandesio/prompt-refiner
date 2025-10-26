from typing import Any, Optional
from pydantic.main import BaseModel
import json
import logging
import re as pyre
import regex

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prompt_refinement")

# Primary extractor using 'regex' recursive pattern (?R)
JSON_BLOCK_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", regex.DOTALL)

# Recursive-ish "{...}" matcher
FENCE_OPEN_RE = pyre.compile(r"^```(?:json)?\s*", pyre.IGNORECASE)
FENCE_CLOSE_RE = pyre.compile(r"\s*```$")


def to_model(model_cls: type[BaseModel], raw, *, step: str = "unknown"):
    """
    Parse model output into a Pydantic model with detailed debug logs on failure.
    Pass step="analyze"/"synthesize"/"evaluate"/"test_case"/"revision" when calling.
    """
    try:
        if isinstance(raw, model_cls):
            return raw
        if isinstance(raw, dict):
            return model_cls.model_validate(raw)
        if isinstance(raw, str):
            cleaned = _clean_json_text(raw)
            try:
                data = json.loads(cleaned)
            except Exception as e_json:
                log.warning(
                    "JSON parse failed | step=%s model=%s\nRAW: %s\nCLEANED: %s\nERROR: %s",
                    step,
                    model_cls.__name__,
                    _truncate(raw),
                    _truncate(cleaned),
                    e_json,
                )
                raise
            try:
                return model_cls.model_validate(data)
            except Exception as e_val:
                log.warning(
                    "Model validate failed | step=%s model=%s\nDATA: %s\nERROR: %s",
                    step,
                    model_cls.__name__,
                    _truncate(data),
                    e_val,
                )
                raise

        log.warning(
            "Unsupported type | step=%s model=%s type=%s",
            step,
            model_cls.__name__,
            type(raw),
        )
        raise ValueError(f"Cannot coerce to {model_cls.__name__}: {type(raw)}")
    except Exception:
        # Re-raise so caller fallbacks (plain text) still run
        raise


def _clean_json_text(s: str) -> str:
    """
    1) strip code fences
    2) if not a single {...}, extract the largest balanced JSON object
    3) normalize quotes + trailing commas
    """
    original = s
    s = _strip_code_fences(original)

    st = s.strip()
    if not (st.startswith("{") and st.endswith("}")):
        cand = _largest_json_object(s)
        if cand:
            s = cand

    # normalize curly quotes and trailing commas
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = pyre.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = FENCE_OPEN_RE.sub("", s)
        s = FENCE_CLOSE_RE.sub("", s)
    return s.strip()


def _largest_json_object(text: str) -> Optional[str]:
    # try recursive-regex first, then brace-stack fallback
    try:
        cand = _extract_largest_json_regex(text)
        if cand:
            return cand
    except Exception as e:
        log.debug("regex (?R) extract failed; falling back. err=%s", e)
    return _extract_largest_json_fallback(text)


# Fallback extractor (brace stack) in case regex fails for some reason
def _extract_largest_json_fallback(text: str) -> Optional[str]:
    best_start = best_end = -1
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
            if len(stack) == 1:
                start_idx = i
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    end_idx = i + 1
                    if (best_end - best_start) < (end_idx - start_idx):
                        best_start, best_end = start_idx, end_idx
                    start_idx = None
    if best_start != -1:
        return text[best_start:best_end]
    return None


def truncate(v: Any, n: int = 800) -> str:
    return _truncate(v, n)


def _truncate(v: Any, n: int = 800) -> str:
    try:
        if isinstance(v, (dict, list)):
            t = json.dumps(v, ensure_ascii=False)
        else:
            t = str(v)
    except Exception:
        t = repr(v)
    return (t[:n] + " …[truncated]") if len(t) > n else t


def _extract_largest_json_regex(text: str) -> Optional[str]:
    matches = list(JSON_BLOCK_RE.finditer(text))
    if not matches:
        return None
    # choose longest balanced {...}
    m = max(matches, key=lambda m: (m.end() - m.start()))
    return text[m.start() : m.end()]
