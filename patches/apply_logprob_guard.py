"""Insert the input-logprob span guard into the pinned SGLang source tree."""

from pathlib import Path


target = Path(
    "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py"
)
source = target.read_text()
if "def _check_input_logprob_span" in source:
    raise SystemExit("guard is already present; refusing a double insertion")

anchor = "logger = logging.getLogger(__name__)\n"
if anchor not in source:
    raise SystemExit("logger anchor was not found in tokenizer_manager.py")

guard = Path("/tmp/logprob_guard.py").read_text()
target.write_text(source.replace(anchor, anchor + "\n\n" + guard + "\n", 1))
print("input-logprob guard inserted")
