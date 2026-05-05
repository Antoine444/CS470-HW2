"""Modulo scheduler for the loop.pip variant.

Pipeline of the algorithm:

1. ASAP-schedule BB0 (same logic as schedule_loop).
2. Compute II_res = max(ceil(N_alu/2), N_mult, N_mem, 1) over BB1 body.
3. For II from II_res upward, try a modulo schedule:
   - ASAP-place each body instr at an internal cycle. Placing at cycle c
     reserves kernel slot (c - bb1_start_initial) mod II for that unit
     across all stages.
   - After scheduling, every interloop dep must satisfy
       S(P) + lambda(P) <= S(C) + II.
     Failure -> bump II.
4. actual bb1_start = min body cycle (BB0-induced padding stays outside
   the kernel).  K = (max body cycle - bb1_start) // II + 1.
5. Assemble output bundles: BB0 + padding bundles + II kernel bundles
   (with each body instr placed in kernel slot (c - bb1_start) mod II
   and stage (c - bb1_start) // II) + BB2.
6. The branch goes in the kernel's last bundle's Branch slot, at internal
   cycle bb1_start + K*II - 1.  Its `target` is the kernel's first output
   bundle index (alloc_r will rewrite if it inserts an init bundle).
7. BB2 is scheduled ASAP after the kernel while enforcing post_loop
   constraints for the last dynamic loop iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .parse import (
    Instr, BRANCH_OPS, UNIT_ALU, UNIT_MULT, UNIT_MEM, UNIT_BRANCH,
    latency_of,
)
from .deps import (
    DepAnalysis, LOCAL, LOOP_INVARIANT, POST_LOOP, INTERLOOP,
)
from .schedule_loop import (
    Schedule, _new_bundle, _ensure, _free_slot, _place, _schedule_block,
    SLOT_ALU0, SLOT_ALU1, SLOT_MULT, SLOT_MEM, SLOT_BRANCH,
)


@dataclass
class PipSchedule:
    bundles: list = field(default_factory=list)
    cycle_of: dict[int, int] = field(default_factory=dict)
    slot_of: dict[int, int] = field(default_factory=dict)
    stage_of: dict[int, int] = field(default_factory=dict)
    II: Optional[int] = None
    K: Optional[int] = None                 # number of pipeline stages
    bb1_start_internal: Optional[int] = None  # internal cycle of stage 0 / kernel slot 0
    kernel_start: Optional[int] = None      # output bundle where kernel begins
    bb2_start: Optional[int] = None
    branch: Optional[Instr] = None


def schedule_pip(instrs: list[Instr], da: DepAnalysis) -> PipSchedule:
    sched = PipSchedule()
    addr_to = {i.addr: i for i in instrs}

    # --- BB0 (same ASAP as the simple-loop scheduler).
    _schedule_block(da.bb0, da, sched, start_cycle=0, addr_to=addr_to)

    if not da.bb1:
        return sched

    body   = [i for i in da.bb1 if i.op not in BRANCH_OPS]
    branch = next(i for i in da.bb1 if i.op in BRANCH_OPS)
    branch.op = "loop.pip"  # pip schedule always uses loop.pip regardless of input

    # --- Resource lower bound for II.
    n_alu  = sum(1 for i in body if i.unit == UNIT_ALU)
    n_mult = sum(1 for i in body if i.unit == UNIT_MULT)
    n_mem  = sum(1 for i in body if i.unit == UNIT_MEM)
    II_res = max((n_alu + 1) // 2, n_mult, n_mem, 1)

    initial_bb1_start = len(sched.bundles)

    # --- Iterate II until a valid schedule is found.
    II = II_res
    assignment: Optional[dict[int, tuple[int, int]]] = None
    while True:
        assignment = _try_modulo(body, da, addr_to, sched, II, initial_bb1_start)
        if assignment is not None:
            break
        II += 1
        if II > max(2 * len(body) + 10, 100):
            raise RuntimeError(f"Modulo scheduling failed to converge (II={II})")

    # --- Recompute actual bb1_start and K from the assignment.
    body_cycles = [c for c, _ in assignment.values()]
    if body_cycles:
        bb1_start = min(body_cycles)
        max_body  = max(body_cycles)
    else:
        bb1_start = initial_bb1_start
        max_body  = initial_bb1_start - 1
    K = (max_body - bb1_start) // II + 1 if body_cycles else 1

    # --- Pad bundles between BB0 and the kernel.
    while len(sched.bundles) < bb1_start:
        sched.bundles.append(_new_bundle())

    # --- Reserve II kernel bundles.
    kernel_start = len(sched.bundles)
    for _ in range(II):
        sched.bundles.append(_new_bundle())

    sched.II = II
    sched.K = K
    sched.bb1_start_internal = bb1_start
    sched.kernel_start = kernel_start
    sched.branch = branch

    # --- Place body instrs in kernel.
    for ins in body:
        c, slot = assignment[ins.addr]
        kslot = (c - bb1_start) % II
        stage = (c - bb1_start) // II
        out_idx = kernel_start + kslot
        if sched.bundles[out_idx][slot] is not None:
            raise RuntimeError(f"kernel slot conflict at {out_idx} slot {slot}")
        sched.bundles[out_idx][slot] = ins
        sched.cycle_of[ins.addr] = out_idx
        sched.slot_of[ins.addr] = slot
        sched.stage_of[ins.addr] = stage

    # --- Place branch in last kernel bundle's Branch slot.
    branch_internal = bb1_start + K * II - 1
    branch_kslot = (branch_internal - bb1_start) % II  # = II - 1
    branch_idx = kernel_start + branch_kslot
    if sched.bundles[branch_idx][SLOT_BRANCH] is not None:
        raise RuntimeError("branch slot already occupied")
    sched.bundles[branch_idx][SLOT_BRANCH] = branch
    sched.cycle_of[branch.addr] = branch_idx
    sched.slot_of[branch.addr] = SLOT_BRANCH
    sched.stage_of[branch.addr] = K - 1
    branch.target = kernel_start  # alloc_r may rewrite if it inserts init

    # --- BB2: ASAP after the kernel, respecting post_loop readiness.
    sched.bb2_start = len(sched.bundles)
    _schedule_bb2_pip(da.bb2, da, sched, addr_to, sched.bb2_start)

    return sched


# --- modulo scheduling core ----------------------------------------------

def _try_modulo(
    body: list[Instr],
    da: DepAnalysis,
    addr_to: dict[int, Instr],
    sched: PipSchedule,
    II: int,
    bb1_start: int,
) -> Optional[dict[int, tuple[int, int]]]:
    """Returns {addr: (internal_cycle, slot)} on success, None to try larger II."""
    kernel = [_new_bundle() for _ in range(II)]
    out: dict[int, tuple[int, int]] = {}

    for ins in body:
        if ins.unit is None:
            continue
        earliest = bb1_start
        for d in da.deps_of(ins.addr):
            if d.kind == LOCAL:
                # producer is somewhere else in BB1 body
                if d.producer_addr in out:
                    p_cycle = out[d.producer_addr][0]
                    earliest = max(earliest, p_cycle + latency_of(addr_to[d.producer_addr].op))
            elif d.kind == LOOP_INVARIANT:
                p_cycle = sched.cycle_of.get(d.producer_addr)
                if p_cycle is not None:
                    earliest = max(earliest, p_cycle + latency_of(addr_to[d.producer_addr].op))
            elif d.kind == INTERLOOP and d.interloop_bb0_addr is not None:
                p_cycle = sched.cycle_of.get(d.interloop_bb0_addr)
                if p_cycle is not None:
                    earliest = max(earliest, p_cycle + latency_of(addr_to[d.interloop_bb0_addr].op))

        placed = False
        for delta in range(II):
            c = earliest + delta
            kslot = (c - bb1_start) % II
            free = _free_slot(kernel[kslot], ins.unit)
            if free is not None:
                kernel[kslot][free] = ins
                out[ins.addr] = (c, free)
                placed = True
                break
        if not placed:
            return None  # II infeasible (shouldn't happen if II >= II_res)

    # --- verify interloop deps
    for ins in body:
        for d in da.deps_of(ins.addr):
            if d.kind == INTERLOOP and d.producer_addr in out:
                p_cycle = out[d.producer_addr][0]
                c_cycle = out[ins.addr][0]
                p_lat = latency_of(addr_to[d.producer_addr].op)
                if p_cycle + p_lat > c_cycle + II:
                    return None

    return out


# --- BB2 scheduling -------------------------------------------------------

def _schedule_bb2_pip(
    block: list[Instr],
    da: DepAnalysis,
    sched: PipSchedule,
    addr_to: dict[int, Instr],
    start_cycle: int,
) -> None:
    for ins in block:
        if ins.unit is None:
            continue
        earliest = start_cycle
        for d in da.deps_of(ins.addr):
            if d.kind == LOCAL:
                p_cycle = sched.cycle_of.get(d.producer_addr)
                if p_cycle is not None:
                    earliest = max(earliest, p_cycle + latency_of(addr_to[d.producer_addr].op))
            elif d.kind == POST_LOOP:
                earliest = max(earliest, _post_loop_ready_cycle(d, sched, addr_to, start_cycle))
            # LOOP_INVARIANT and INTERLOOP do not appear for BB2 consumers.
        c = earliest
        while True:
            _ensure(sched, c)
            slot = _free_slot(sched.bundles[c], ins.unit)
            if slot is not None:
                _place(sched, ins, c, slot)
                break
            c += 1


def _post_loop_ready_cycle(
    dep,
    sched: PipSchedule,
    addr_to: dict[int, Instr],
    start_cycle: int,
) -> int:
    """Cycle when a BB2 consumer may read a value from the last loop iteration."""
    if sched.II is None or sched.K is None or sched.kernel_start is None:
        raise RuntimeError("missing pip schedule metadata for post-loop dependency")

    p_addr = dep.producer_addr
    if p_addr is None:
        raise RuntimeError("post-loop dependency is missing a producer")
    if p_addr not in sched.cycle_of:
        raise RuntimeError(f"post-loop producer {p_addr} has no scheduled cycle")
    if p_addr not in sched.stage_of:
        raise RuntimeError(f"post-loop producer {p_addr} has no scheduled stage")
    if p_addr not in addr_to:
        raise RuntimeError(f"post-loop producer {p_addr} is unknown")

    p_cycle = sched.cycle_of[p_addr]
    p_stage = sched.stage_of[p_addr]
    p_kslot = p_cycle - sched.kernel_start

    if not (0 <= p_kslot < sched.II):
        raise RuntimeError(
            f"post-loop producer {p_addr} has invalid kernel slot {p_kslot}"
        )
    if not (0 <= p_stage < sched.K):
        raise RuntimeError(
            f"post-loop producer {p_addr} has invalid stage {p_stage}"
        )

    cycles_before_fallthrough = (
        (sched.K - 1 - p_stage) * sched.II
        + (sched.II - 1 - p_kslot)
    )
    wait_after_fallthrough = max(
        0,
        latency_of(addr_to[p_addr].op) - cycles_before_fallthrough - 1,
    )
    return start_cycle + wait_after_fallthrough
