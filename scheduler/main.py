"""Entry point: ./scheduler/main.py <input.json> <simple.json> <pip.json>"""
import json
import sys

from .parse import parse_program
from .deps import analyze
from .schedule_loop import schedule_simple_loop
from .schedule_pip import schedule_pip
from .alloc import alloc_b, alloc_r
from .emit import write_bundles


def _run(input_path: str, simple_path: str, pip_path: str) -> None:
    raw = json.load(open(input_path))

    # Simple-loop schedule (always uses alloc_b).
    instrs_s = parse_program(raw)
    da_s = analyze(instrs_s)
    s = schedule_simple_loop(instrs_s, da_s)
    alloc_b(instrs_s, da_s, s)
    write_bundles(s.bundles, simple_path)

    # Pip schedule. If there is no loop, it degenerates to the same as simple.
    instrs_p = parse_program(raw)
    da_p = analyze(instrs_p)
    if da_p.bb1:
        p = schedule_pip(instrs_p, da_p)
        alloc_r(instrs_p, da_p, p)
        write_bundles(p.bundles, pip_path)
    else:
        # No loop -> reuse the simple schedule (mirrors the references).
        instrs_p2 = parse_program(raw)
        da_p2 = analyze(instrs_p2)
        sp = schedule_simple_loop(instrs_p2, da_p2)
        alloc_b(instrs_p2, da_p2, sp)
        write_bundles(sp.bundles, pip_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: main.py <input.json> <simple.json> <pip.json>", file=sys.stderr)
        sys.exit(1)
    _run(sys.argv[1], sys.argv[2], sys.argv[3])
