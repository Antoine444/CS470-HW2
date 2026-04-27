"""True (RAW) dependency analysis for VLIW470 scheduling.

Per the handout (§3.2), we track only true / read-after-write dependencies.
Anti and output dependencies are eliminated by register allocation.

Each operand read by an instruction is classified into exactly one of:

  local            producer in the same BB as consumer
  interloop        consumer in BB1; producer in BB1 (reaches via previous iter).
                   Optionally also a BB0 producer that supplies the first iter.
  loop_invariant   consumer in BB1; only producer is in BB0 (no BB1 writer).
  post_loop        consumer in BB2; producer is in BB1 (last iteration's value).
  undefined        no producer reaches this consumer (read of an unwritten reg).

Single-source assumption from the handout: for any register r there is at most
one producer per BB, so each operand classification points at one (or in the
interloop case, two) producer instructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .parse import Instr, split_bbs


# Dependency kinds.
LOCAL          = "local"
INTERLOOP      = "interloop"
LOOP_INVARIANT = "loop_invariant"
POST_LOOP      = "post_loop"
UNDEFINED      = "undefined"


@dataclass
class OperandDep:
    """One source-operand dependency for a consumer instruction."""
    consumer_addr: int          # addr of the consuming instruction
    reg: str                    # the source register name as it appears in the input
    kind: str                   # one of the *_ kind constants above
    producer_addr: Optional[int] = None
    # For INTERLOOP: optional BB0 producer that supplies the value in the first iter.
    interloop_bb0_addr: Optional[int] = None


@dataclass
class DepAnalysis:
    instrs: list[Instr]
    bb0: list[Instr]
    bb1: list[Instr]
    bb2: list[Instr]
    by_consumer: dict[int, list[OperandDep]] = field(default_factory=dict)

    def deps_of(self, addr: int) -> list[OperandDep]:
        return self.by_consumer.get(addr, [])

    def block_of(self, addr: int) -> str:
        if any(i.addr == addr for i in self.bb0): return "BB0"
        if any(i.addr == addr for i in self.bb1): return "BB1"
        if any(i.addr == addr for i in self.bb2): return "BB2"
        raise KeyError(addr)


# ---------------------------------------------------------------------------

def analyze(instrs: list[Instr]) -> DepAnalysis:
    bb0, bb1, bb2 = split_bbs(instrs)
    da = DepAnalysis(instrs=instrs, bb0=bb0, bb1=bb1, bb2=bb2)

    # Latest writer of each register, scanned over the entire BB.
    last_bb0_writer: dict[str, int] = {}
    for ins in bb0:
        w = ins.writes()
        if w is not None:
            last_bb0_writer[w] = ins.addr

    last_bb1_writer: dict[str, int] = {}
    for ins in bb1:
        w = ins.writes()
        if w is not None:
            last_bb1_writer[w] = ins.addr

    bb0_addrs = {i.addr for i in bb0}
    bb1_addrs = {i.addr for i in bb1}
    bb2_addrs = {i.addr for i in bb2}

    for ins in instrs:
        ops: list[OperandDep] = []
        for r in ins.reads():
            if ins.addr in bb0_addrs:
                ops.append(_classify_bb0(ins, r, bb0))
            elif ins.addr in bb1_addrs:
                ops.append(_classify_bb1(ins, r, bb1, last_bb1_writer, last_bb0_writer))
            else:
                ops.append(_classify_bb2(ins, r, bb2, last_bb1_writer, last_bb0_writer))
        da.by_consumer[ins.addr] = ops

    return da


# ---------------------------------------------------------------------------

def _last_writer_before(bb: list[Instr], reg: str, before_addr: int) -> Optional[int]:
    """Latest producer of `reg` in `bb` strictly before `before_addr` in program order."""
    last: Optional[int] = None
    for ins in bb:
        if ins.addr >= before_addr:
            break
        if ins.writes() == reg:
            last = ins.addr
    return last


def _classify_bb0(consumer: Instr, reg: str, bb0: list[Instr]) -> OperandDep:
    p = _last_writer_before(bb0, reg, consumer.addr)
    if p is not None:
        return OperandDep(consumer.addr, reg, LOCAL, producer_addr=p)
    return OperandDep(consumer.addr, reg, UNDEFINED)


def _classify_bb1(
    consumer: Instr,
    reg: str,
    bb1: list[Instr],
    last_bb1_writer: dict[str, int],
    last_bb0_writer: dict[str, int],
) -> OperandDep:
    # local: latest BB1 writer strictly before consumer
    local_p = _last_writer_before(bb1, reg, consumer.addr)
    if local_p is not None:
        return OperandDep(consumer.addr, reg, LOCAL, producer_addr=local_p)

    # interloop: any BB1 writer of reg (necessarily at or after consumer in program
    # order — supplies the value via the previous iteration). If BB0 also writes
    # reg, it supplies the first-iteration value.
    bb1_p = last_bb1_writer.get(reg)
    bb0_p = last_bb0_writer.get(reg)
    if bb1_p is not None:
        return OperandDep(
            consumer.addr, reg, INTERLOOP,
            producer_addr=bb1_p,
            interloop_bb0_addr=bb0_p,
        )

    if bb0_p is not None:
        return OperandDep(consumer.addr, reg, LOOP_INVARIANT, producer_addr=bb0_p)

    return OperandDep(consumer.addr, reg, UNDEFINED)


def _classify_bb2(
    consumer: Instr,
    reg: str,
    bb2: list[Instr],
    last_bb1_writer: dict[str, int],
    last_bb0_writer: dict[str, int],
) -> OperandDep:
    # local within BB2: latest BB2 writer before consumer
    p = _last_writer_before(bb2, reg, consumer.addr)
    if p is not None:
        return OperandDep(consumer.addr, reg, LOCAL, producer_addr=p)

    # post-loop: any BB1 writer (last iteration's value)
    bb1_p = last_bb1_writer.get(reg)
    if bb1_p is not None:
        return OperandDep(consumer.addr, reg, POST_LOOP, producer_addr=bb1_p)

    # value persists from BB0
    bb0_p = last_bb0_writer.get(reg)
    if bb0_p is not None:
        return OperandDep(consumer.addr, reg, LOCAL, producer_addr=bb0_p)

    return OperandDep(consumer.addr, reg, UNDEFINED)
