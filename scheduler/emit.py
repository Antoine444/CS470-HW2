"""Render Instr objects back to canonical strings and write bundle JSON.

Output convention (matches the handout reference files):
- Each bundle is a list of exactly 5 strings: [ALU0, ALU1, Mult, Mem, Branch].
- Empty slots are " nop" (leading space).
- Non-predicated instructions are emitted with a single leading space.
- Predicated instructions start with "(pN) " (no leading space).

compare.py strips whitespace, so the leading-space convention is purely cosmetic
to make manual diffs against the reference readable.
"""
from __future__ import annotations

import json
from typing import Optional

from .parse import Instr


NOP = "nop"


def format_instr(ins: Optional[Instr]) -> str:
    if ins is None:
        return NOP

    prefix = f"({ins.pred}) " if ins.pred else " "
    op = ins.op

    if op == NOP:
        return prefix + NOP

    if op in ("add", "sub", "mulu"):
        return f"{prefix}{op} {ins.dest}, {ins.src_a}, {ins.src_b}"

    if op == "addi":
        return f"{prefix}addi {ins.dest}, {ins.src_a}, {ins.imm}"

    if op == "ld":
        return f"{prefix}ld {ins.dest}, {ins.imm}({ins.mem_base})"

    if op == "st":
        return f"{prefix}st {ins.src_a}, {ins.imm}({ins.mem_base})"

    if op == "mov":
        if ins.dest in ("LC", "EC"):
            return f"{prefix}mov {ins.dest}, {ins.imm}"
        if ins.dest and ins.dest.startswith("p"):
            return f"{prefix}mov {ins.dest}, {'true' if ins.bool_val else 'false'}"
        if ins.src_a is not None:
            return f"{prefix}mov {ins.dest}, {ins.src_a}"
        return f"{prefix}mov {ins.dest}, {ins.imm}"

    if op in ("loop", "loop.pip"):
        return f"{prefix}{op} {ins.target}"

    raise ValueError(f"Cannot format instruction: {ins}")


Bundle = list  # alias: list of length 5, each entry None | Instr


def bundle_to_strings(bundle: Bundle) -> list[str]:
    if len(bundle) != 5:
        raise ValueError(f"Bundle must have 5 slots, got {len(bundle)}")
    return [format_instr(slot) for slot in bundle]


def write_bundles(bundles: list[Bundle], path: str) -> None:
    out = [bundle_to_strings(b) for b in bundles]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
