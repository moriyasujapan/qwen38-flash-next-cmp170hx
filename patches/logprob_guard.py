# Reject input-logprob requests whose full-vocabulary rows can exhaust VRAM.
# This block is inserted into tokenizer_manager.py, which already imports os.
_MAX_INPUT_LOGPROB_TOKENS = int(
    os.environ.get("SGLANG_MAX_INPUT_LOGPROB_TOKENS", "1024")
)


def _check_input_logprob_span(
    return_logprob,
    logprob_start_len,
    n_input_tokens: int,
    cap: int = None,
) -> None:
    """Raise ValueError when too many prompt positions would be scored."""
    cap = _MAX_INPUT_LOGPROB_TOKENS if cap is None else cap
    if not return_logprob or cap <= 0:
        return
    if isinstance(logprob_start_len, (list, tuple)):
        starts = [item for item in logprob_start_len if item is not None]
        if not starts:
            return
        logprob_start_len = min(starts)
    if logprob_start_len is None or logprob_start_len < 0:
        return

    scored = n_input_tokens - int(logprob_start_len)
    if scored > cap:
        raise ValueError(
            f"Input logprobs were requested for {scored} prompt positions "
            f"(logprob_start_len={int(logprob_start_len)}, "
            f"prompt={n_input_tokens} tokens). This server allows at most "
            f"{cap} scored prompt positions per request. Set "
            f"logprob_start_len >= {n_input_tokens - cap}, or score the text "
            f"in windows of <= {cap} positions. Server override: "
            "SGLANG_MAX_INPUT_LOGPROB_TOKENS."
        )
