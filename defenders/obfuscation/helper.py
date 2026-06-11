def normalize_input(raw_input: str | bytes) -> str:
    if isinstance(raw_input, bytes):
        return raw_input.decode("utf-8", errors="replace")
    return raw_input