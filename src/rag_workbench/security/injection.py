UNTRUSTED_CONTEXT_OPEN = "<untrusted_retrieved_context>"
UNTRUSTED_CONTEXT_CLOSE = "</untrusted_retrieved_context>"


def wrap_untrusted_context(content: str) -> str:
    escaped = content.replace(UNTRUSTED_CONTEXT_CLOSE, "&lt;/untrusted_retrieved_context&gt;")
    return f"{UNTRUSTED_CONTEXT_OPEN}\n{escaped}\n{UNTRUSTED_CONTEXT_CLOSE}"
