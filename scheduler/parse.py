"""Parse VLIW470 input programs into structured Instr objects.

Input syntax (one instruction per JSON list element):

    nop
    add  dest, srcA, srcB
    addi dest, srcA, imm
    sub  dest, srcA, srcB
    mulu dest, srcA, srcB
    ld   dest, imm(base)
    st   src,  imm(base)
    mov  LC,    imm
    mov  EC,    imm
    mov  pN,    true|false
    mov  dest,  imm
    mov  dest,  src
    loop      target
    loop.pip  target

Any instruction except loop / loop.pip can be predicated:  (pN) <instr>

Immediates may be hex (0x...), decimal, or negative; they are normalised to
signed decimal because compare.py does string equality on the slot text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --- execution units / latencies ------------------------------------------

ALU_OPS    = {"add", "addi", "sub", "mov"}
MULT_OPS   = {"mulu"}
MEM_OPS    = {"ld", "st"}
BRANCH_OPS = {"loop", "loop.pip"}

UNIT_ALU, UNIT_MULT, UNIT_MEM, UNIT_BRANCH = "alu", "mult", "mem", "branch"

LATENCY = {"mulu": 3}  # everything else: 1


def latency_of(op: str) -> int:
    return LATENCY.get(op, 1)


def unit_of(op: str) -> Optional[str]:
    if op in ALU_OPS:    return UNIT_ALU
    if op in MULT_OPS:   return UNIT_MULT
    if op in MEM_OPS:    return UNIT_MEM
    if op in BRANCH_OPS: return UNIT_BRANCH
    if op == "nop":      return None
    raise ValueError(f"Unknown opcode: {op!r}")


# --- Instr data class ------------------------------------------------------

@dataclass
class Instr:
    addr: int                                # original program index (line in input)
    op: str                                  # opcode
    pred: Optional[str] = None               # predicate register name, e.g. "p32"
    dest: Optional[str] = None               # dest reg name (or "LC", "EC", "pN")
    src_a: Optional[str] = None              # first source / value-being-stored / mov src
    src_b: Optional[str] = None              # second source (3-operand ALU / mulu)
    imm: Optional[int] = None                # immediate (decimal int)
    mem_base: Optional[str] = None           # base reg for ld/st
    target: Optional[int] = None             # loop / loop.pip target (bundle index)
    bool_val: Optional[bool] = None          # true/false for mov pN, ...

    @property
    def unit(self) -> Optional[str]:
        return unit_of(self.op)

    @property
    def latency(self) -> int:
        return latency_of(self.op)

    def reads(self) -> list[str]:
        """Source registers (xN only) read by this instruction."""
        out: list[str] = []
        for r in (self.src_a, self.src_b, self.mem_base):
            if r is not None and _is_gp(r):
                out.append(r)
        return out

    def writes(self) -> Optional[str]:
        """Destination GP register written, or None."""
        if self.dest is not None and _is_gp(self.dest):
            return self.dest
        return None


# --- regexes ---------------------------------------------------------------

_PRED_RE = re.compile(r"^\s*\(\s*(p\d+)\s*\)\s*(.*)$")
_MEM_RE  = re.compile(r"^\s*(-?(?:0[xX][0-9a-fA-F]+|\d+))\s*\(\s*(\w+)\s*\)\s*$")
_GP_RE   = re.compile(r"^x(\d+)$")


def _is_gp(s: str) -> bool:
    return bool(_GP_RE.match(s))


def _parse_imm(s: str) -> int:
    """Parse a numeric literal: decimal, hex (0x...), or negative."""
    return int(s.strip(), 0)


# --- instruction parser ----------------------------------------------------

def parse_line(line: str, addr: int) -> Instr:
    s = line.strip()

    pred: Optional[str] = None
    m = _PRED_RE.match(s)
    if m:
        pred = m.group(1)
        s = m.group(2).strip()

    parts = s.split(None, 1)
    if not parts:
        raise ValueError(f"Empty instruction at addr {addr}")
    op = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    args = [a.strip() for a in rest.split(",")] if rest else []

    ins = Instr(addr=addr, op=op, pred=pred)

    if op == "nop":
        return ins

    if op in ("add", "sub", "mulu"):
        _expect(args, 3, line)
        ins.dest, ins.src_a, ins.src_b = args[0], args[1], args[2]
        return ins

    if op == "addi":
        _expect(args, 3, line)
        ins.dest, ins.src_a = args[0], args[1]
        ins.imm = _parse_imm(args[2])
        return ins

    if op == "ld":
        _expect(args, 2, line)
        ins.dest = args[0]
        ins.imm, ins.mem_base = _split_mem(args[1], line)
        return ins

    if op == "st":
        _expect(args, 2, line)
        ins.src_a = args[0]
        ins.imm, ins.mem_base = _split_mem(args[1], line)
        return ins

    if op == "mov":
        _expect(args, 2, line)
        d, v = args[0], args[1]
        if d.upper() in ("LC", "EC"):
            ins.dest = d.upper()
            ins.imm = _parse_imm(v)
        elif re.match(r"^p\d+$", d):
            ins.dest = d
            lv = v.lower()
            if lv == "true":    ins.bool_val = True
            elif lv == "false": ins.bool_val = False
            else:
                raise ValueError(f"Bad predicate mov: {line!r}")
        else:
            ins.dest = d
            if _is_gp(v):
                ins.src_a = v
            else:
                ins.imm = _parse_imm(v)
        return ins

    if op in ("loop", "loop.pip"):
        _expect(args, 1, line)
        ins.target = _parse_imm(args[0])
        return ins

    raise ValueError(f"Unknown opcode in line {addr}: {line!r}")


def _expect(args: list, n: int, line: str) -> None:
    if len(args) != n:
        raise ValueError(f"Expected {n} args, got {len(args)} in line {line!r}")


def _split_mem(s: str, line: str) -> tuple[int, str]:
    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"Bad memory operand {s!r} in {line!r}")
    return _parse_imm(m.group(1)), m.group(2)


# --- program / basic-block helpers ----------------------------------------

def parse_program(lines: list[str]) -> list[Instr]:
    return [parse_line(l, i) for i, l in enumerate(lines)]


def find_loop_index(instrs: list[Instr]) -> Optional[int]:
    for i, ins in enumerate(instrs):
        if ins.op in BRANCH_OPS:
            return i
    return None


def split_bbs(instrs: list[Instr]) -> tuple[list[Instr], list[Instr], list[Instr]]:
    """Return (BB0, BB1, BB2). If no loop instruction, BB1 = BB2 = []."""
    li = find_loop_index(instrs)
    if li is None:
        return list(instrs), [], []
    target = instrs[li].target
    if target is None or not (0 <= target <= li):
        raise ValueError(f"Loop at addr {li} has invalid target {target!r}")
    bb0 = instrs[:target]
    bb1 = instrs[target:li + 1]
    bb2 = instrs[li + 1:]
    return bb0, bb1, bb2
