"""ASAP scheduler for the simple-loop variant (`loop` instruction).

Three basic blocks (BB0, BB1, BB2) are scheduled in program order, each
independently. Within a block, instructions are placed greedily at the earliest
cycle that satisfies their already-scheduled RAW dependencies and where the
required execution unit's slot is free.

For BB1, the loop branch is placed in the Branch slot of the last bundle. We
then verify that every interloop dependency satisfies

    S(P) + lambda(P) <= S(C) + II        (II = #bundles in BB1)

If violated, we push the branch down by appending an empty bundle (II += 1) and
re-check.

The branch's `target` field is rewritten to the BB1 start bundle index so that
emit produces e.g. " loop 1" pointing at the actual loop entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .parse import (
    Instr, BRANCH_OPS, UNIT_ALU, UNIT_MULT, UNIT_MEM, UNIT_BRANCH,
    latency_of,
)
from .deps import DepAnalysis, LOCAL, LOOP_INVARIANT, POST_LOOP, INTERLOOP


SLOT_ALU0, SLOT_ALU1, SLOT_MULT, SLOT_MEM, SLOT_BRANCH = 0, 1, 2, 3, 4

Bundle = list  # length 5; each entry is Optional[Instr]


@dataclass
class Schedule:
    bundles: list                                       # list[Bundle]
    cycle_of: dict[int, int] = field(default_factory=dict)
    slot_of: dict[int, int]  = field(default_factory=dict)
    bb1_start: Optional[int] = None
    bb1_end:   Optional[int] = None                      # bundle holding the loop branch
    bb2_start: Optional[int] = None

    @property
    def II(self) -> Optional[int]:
        if self.bb1_start is None or self.bb1_end is None:
            return None
        return self.bb1_end - self.bb1_start + 1


# --- core helpers ---------------------------------------------------------

def _new_bundle() -> Bundle:
    return [None] * 5


def _ensure(sched: Schedule, cycle: int) -> None:
    while len(sched.bundles) <= cycle:
        sched.bundles.append(_new_bundle())


def _free_slot(bundle: Bundle, unit: str) -> Optional[int]:
    if unit == UNIT_ALU:
        if bundle[SLOT_ALU0] is None: return SLOT_ALU0
        if bundle[SLOT_ALU1] is None: return SLOT_ALU1
        return None
    if unit == UNIT_MULT:   return SLOT_MULT   if bundle[SLOT_MULT]   is None else None
    if unit == UNIT_MEM:    return SLOT_MEM    if bundle[SLOT_MEM]    is None else None
    if unit == UNIT_BRANCH: return SLOT_BRANCH if bundle[SLOT_BRANCH] is None else None
    return None


def _place(sched: Schedule, ins: Instr, cycle: int, slot: int) -> None:
    _ensure(sched, cycle)
    if sched.bundles[cycle][slot] is not None:
        raise RuntimeError(f"slot {slot} at cycle {cycle} already taken")
    sched.bundles[cycle][slot] = ins
    sched.cycle_of[ins.addr] = cycle
    sched.slot_of[ins.addr] = slot


def _earliest_for(ins: Instr, da: DepAnalysis, sched: Schedule, addr_to: dict[int, Instr]) -> int:
    """Min cycle for `ins` that respects every already-scheduled producer.

    Wrap-around interloop deps (the BB1 prev-iter producer) are checked
    separately via the II equation after the whole block is scheduled.
    """
    earliest = 0
    for d in da.deps_of(ins.addr):
        prods = []
        if d.kind in (LOCAL, LOOP_INVARIANT, POST_LOOP):
            prods.append(d.producer_addr)
        elif d.kind == INTERLOOP and d.interloop_bb0_addr is not None:
            prods.append(d.interloop_bb0_addr)
        for paddr in prods:
            p_cycle = sched.cycle_of.get(paddr)
            if p_cycle is None:
                continue
            earliest = max(earliest, p_cycle + latency_of(addr_to[paddr].op))
    return earliest


def _schedule_block(
    block: list[Instr],
    da: DepAnalysis,
    sched: Schedule,
    start_cycle: int,
    addr_to: dict[int, Instr],
) -> None:
    for ins in block:
        if ins.unit is None:
            continue  # explicit nop input — skip, output bundles fill empty slots later
        earliest = max(start_cycle, _earliest_for(ins, da, sched, addr_to))
        c = earliest
        while True:
            _ensure(sched, c)
            slot = _free_slot(sched.bundles[c], ins.unit)
            if slot is not None:
                _place(sched, ins, c, slot)
                break
            c += 1


# --- top-level ------------------------------------------------------------

def schedule_simple_loop(instrs: list[Instr], da: DepAnalysis) -> Schedule:
    sched = Schedule(bundles=[])
    addr_to = {i.addr: i for i in instrs}

    # --- BB0
    _schedule_block(da.bb0, da, sched, start_cycle=0, addr_to=addr_to)

    if not da.bb1:
        return sched

    # --- BB1 (body then branch at last bundle's Branch slot)
    initial_start = len(sched.bundles)

    body   = [i for i in da.bb1 if i.op not in BRANCH_OPS]
    branch = next(i for i in da.bb1 if i.op in BRANCH_OPS)

    _schedule_block(body, da, sched, start_cycle=initial_start, addr_to=addr_to)

    # Empty bundles inserted before the first body instr (e.g. waiting on a
    # long-latency BB0 producer) belong before the loop, not inside it. The
    # handout: "add the empty delay bundles before/after the loop, but never
    # within the loop body". So the actual bb1_start is the first body cycle.
    if body:
        body_cycles = [sched.cycle_of[i.addr] for i in body]
        bb1_start = min(body_cycles)
        last_used = max(body_cycles)
    else:
        bb1_start = initial_start
        last_used = initial_start - 1

    sched.bb1_start = bb1_start
    bb1_end = max(last_used, bb1_start)
    _place(sched, branch, bb1_end, SLOT_BRANCH)

    _resolve_interloop(sched, da, bb1_start, branch, addr_to)

    sched.bb1_end = sched.cycle_of[branch.addr]
    branch.target = bb1_start

    # --- BB2
    bb2_start = len(sched.bundles)
    sched.bb2_start = bb2_start
    _schedule_block(da.bb2, da, sched, start_cycle=bb2_start, addr_to=addr_to)

    return sched


def _resolve_interloop(
    sched: Schedule,
    da: DepAnalysis,
    bb1_start: int,
    branch: Instr,
    addr_to: dict[int, Instr],
) -> None:
    """Push the loop branch down (II += 1) while any interloop dep is violated."""
    while True:
        II = sched.cycle_of[branch.addr] - bb1_start + 1
        violated_ins = None
        for ins in da.bb1:
            for d in da.deps_of(ins.addr):
                if d.kind != INTERLOOP:
                    continue
                p = addr_to[d.producer_addr]
                if sched.cycle_of[p.addr] + latency_of(p.op) > sched.cycle_of[ins.addr] + II:
                    violated_ins = ins
                    break
            if violated_ins is not None:
                break
        if violated_ins is None:
            return

        # Push branch down by 1.
        cur = sched.cycle_of[branch.addr]
        sched.bundles[cur][SLOT_BRANCH] = None
        sched.bundles.append(_new_bundle())
        new_cycle = cur + 1
        sched.bundles[new_cycle][SLOT_BRANCH] = branch
        sched.cycle_of[branch.addr] = new_cycle
