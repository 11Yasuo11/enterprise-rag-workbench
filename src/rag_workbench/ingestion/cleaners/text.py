import re

SPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    lines = [SPACE.sub(" ", line).strip() for line in text.replace("\x00", "").splitlines()]
    return BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
