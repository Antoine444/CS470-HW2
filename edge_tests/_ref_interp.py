"""Pure-Python sequential interpreter for VLIW470 flat input programs.

Executes the input.json instruction list with full results available immediately
(no pipeline latency, no rotating registers). Used as a functional oracle by
edge_tests/validate.py.

Supports: add, addi, sub, mulu, mov (reg|imm|LC|EC|pN), ld, st, loop, loop.pip.
Treats loop / loop.pip as plain decrement-LC-and-branch on LC > 0; the
prologue/kernel/epilogue / RBB / predicate plumbing is irrelevant in a flat
program because the input never reads a rotating register or predicate.
"""
from __future__ import annotations

import json
import re
from typing import Any

MASK64 = 0xFFFFFFFFFFFFFFFF
_PRED_RE = re.compile(r"^\s*\(\s*p(\d+)\s*\)\s*(.*)$")
_MEM_RE = re.compile(r"^\s*(-?(?:0[xX][0-9a-fA-F]+|\d+))\s*\(\s*x(\d+)\s*\)\s*$")


def _imm(s: str) -> int:
    return int(s.strip(), 0)


class RefState:
    def __init__(self) -> None:
        self.regs = [0] * 96  # x0..x95 (we only ever use x0..x63 in inputs)
        self.preds = [False] * 96
        self.LC = 0
        self.EC = 0
        self.mem: dict[int, int] = {}

    def load_memory(self, init: dict[str, int]) -> None:
        for k, v in init.items():
            addr = int(k, 16) if str(k).startswith("0x") else int(k)
            self.mem[addr] = int(v) & MASK64

    def read_mem(self, addr: int) -> int:
        return self.mem.get(addr, 0)


def _parse(line: str) -> tuple[bool | None, str, list[str]]:
    """Returns (predicate-bool-or-None, opcode, operand-list).

    Predicate bool is None when there's no predicate, otherwise the value of
    the predicate register at decode time. We pass None / True / False up so
    the caller can decide whether to skip the instruction.
    """
    s = line.strip()
    pred_idx = None
    m = _PRED_RE.match(s)
    if m:
        pred_idx = int(m.group(1))
        s = m.group(2).strip()
    parts = s.split(None, 1)
    op = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    args = [a.strip() for a in rest.split(",")] if rest else []
    return pred_idx, op, args


def execute(instructions: list[str], memory_init: dict[str, int] | None = None,
            max_cycles: int = 10_000_000) -> RefState:
    st = RefState()
    if memory_init:
        st.load_memory(memory_init)

    pc = 0
    steps = 0
    while pc < len(instructions):
        steps += 1
        if steps > max_cycles:
            raise RuntimeError(f"interpreter ran {max_cycles} steps; suspect infinite loop")
        line = instructions[pc]
        pred_idx, op, args = _parse(line)

        if pred_idx is not None and not st.preds[pred_idx]:
            pc += 1
            continue

        if op == "nop":
            pc += 1
            continue

        if op == "add":
            d, a, b = args
            st.regs[int(d[1:])] = (st.regs[int(a[1:])] + st.regs[int(b[1:])]) & MASK64
            pc += 1
        elif op == "addi":
            d, a, k = args
            st.regs[int(d[1:])] = (st.regs[int(a[1:])] + _imm(k)) & MASK64
            pc += 1
        elif op == "sub":
            d, a, b = args
            v = st.regs[int(a[1:])] - st.regs[int(b[1:])]
            st.regs[int(d[1:])] = v & MASK64
            pc += 1
        elif op == "mulu":
            d, a, b = args
            st.regs[int(d[1:])] = (st.regs[int(a[1:])] * st.regs[int(b[1:])]) & MASK64
            pc += 1
        elif op == "mov":
            d, v = args
            du = d.upper()
            if du == "LC":
                st.LC = _imm(v) & MASK64
            elif du == "EC":
                st.EC = _imm(v) & MASK64
            elif d.startswith("p"):
                idx = int(d[1:])
                if v.lower() == "true":
                    st.preds[idx] = True
                elif v.lower() == "false":
                    st.preds[idx] = False
                else:
                    raise ValueError(f"bad mov pred: {line!r}")
            elif d.startswith("x"):
                idx = int(d[1:])
                if v.startswith("x"):
                    st.regs[idx] = st.regs[int(v[1:])]
                else:
                    st.regs[idx] = _imm(v) & MASK64
            else:
                raise ValueError(f"bad mov dest: {line!r}")
            pc += 1
        elif op == "ld":
            d, mem = args
            m = _MEM_RE.match(mem)
            if not m:
                raise ValueError(f"bad ld mem operand: {line!r}")
            offset, base = _imm(m.group(1)), int(m.group(2))
            addr = (st.regs[base] + offset) & MASK64
            st.regs[int(d[1:])] = st.read_mem(addr)
            pc += 1
        elif op == "st":
            src, mem = args
            m = _MEM_RE.match(mem)
            if not m:
                raise ValueError(f"bad st mem operand: {line!r}")
            offset, base = _imm(m.group(1)), int(m.group(2))
            addr = (st.regs[base] + offset) & MASK64
            st.mem[addr] = st.regs[int(src[1:])] & MASK64
            pc += 1
        elif op in ("loop", "loop.pip"):
            target = _imm(args[0])
            if st.LC > 0:
                st.LC = (st.LC - 1) & MASK64
                pc = target
            else:
                pc += 1
        else:
            raise ValueError(f"unknown opcode in flat input: {op!r} ({line!r})")

    return st


def run_file(input_path: str, memory_path: str | None = None) -> RefState:
    with open(input_path) as f:
        instrs: list[str] = json.load(f)
    mem_init: dict[str, int] | None = None
    if memory_path:
        with open(memory_path) as f:
            mem_init = json.load(f)
    return execute(instrs, mem_init)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--memory")
    args = ap.parse_args()
    s = run_file(args.input, args.memory)
    out: dict[str, Any] = {
        "regs": {f"x{i}": s.regs[i] for i in range(32) if s.regs[i] != 0},
        "memory": {hex(k): v for k, v in sorted(s.mem.items())},
        "LC": s.LC,
        "EC": s.EC,
    }
    print(json.dumps(out, indent=2))
