"""Register allocation for the simple-loop schedule (alloc_b).

Phases (§3.3.1 of the handout):

  1. Walk the schedule in (cycle, slot) order. Each instruction that produces a
     GP register value gets a fresh name x1, x2, ....
  2. Rewrite each operand to the producer's new name, using the dependency
     analysis. Interloop operands with both BB0 and BB1 producers use the BB0
     producer's name; the bridging mov in phase 3 keeps the value alive across
     iterations.
  3. For every (BB0, BB1) interloop pair, insert `mov BB0_reg, BB1_reg` in the
     loop branch's bundle. If the bundle has no free ALU slot or the mov would
     read the BB1 producer before its result is available, push the branch
     down by one bundle and retry.
  4. Operands that have no producer (UNDEFINED reads) get fresh unused
     registers, assigned in scheduling order.
"""
from __future__ import annotations

from typing import Optional

from .parse import Instr, latency_of, BRANCH_OPS, _is_gp
from .deps import (
    DepAnalysis, OperandDep,
    LOCAL, INTERLOOP, LOOP_INVARIANT, POST_LOOP, UNDEFINED,
)
from .schedule_loop import Schedule, SLOT_ALU0, SLOT_ALU1, SLOT_BRANCH
from .schedule_pip import PipSchedule


_OPERAND_FIELDS = ("src_a", "src_b", "mem_base")


def alloc_b(instrs: list[Instr], da: DepAnalysis, sched: Schedule) -> None:
    """Allocate registers for a simple-loop schedule. Mutates instrs and sched."""

    addr_to = {i.addr: i for i in instrs}

    # phase 1 -- fresh dests in scheduling order
    sched_order = sorted(
        (i for i in instrs if i.addr in sched.cycle_of),
        key=lambda i: (sched.cycle_of[i.addr], sched.slot_of[i.addr]),
    )
    new_dest: dict[int, str] = {}
    next_reg = 1
    for ins in sched_order:
        if ins.writes() is not None:
            new_dest[ins.addr] = f"x{next_reg}"
            next_reg += 1
    for ins in sched_order:
        if ins.addr in new_dest:
            ins.dest = new_dest[ins.addr]

    # phase 2 -- rewrite operands using dependency producers
    for ins in instrs:
        worklist = _operand_worklist(ins)
        for field, dep in zip(worklist, da.deps_of(ins.addr)):
            new = _resolve_operand(dep, new_dest)
            if new is not None:
                setattr(ins, field, new)

    # phase 3 -- bridging movs for interloop deps
    if da.bb1:
        branch = next(i for i in da.bb1 if i.op in BRANCH_OPS)
        _insert_bridges(da, sched, addr_to, new_dest, branch)
        branch.target = sched.bb1_start

    # phase 4 -- undefined operands get fresh regs (sched order)
    for ins in sched_order:
        worklist = _operand_worklist(ins)
        for field, dep in zip(worklist, da.deps_of(ins.addr)):
            if dep.kind == UNDEFINED:
                setattr(ins, field, f"x{next_reg}")
                next_reg += 1


# --- helpers --------------------------------------------------------------

def _operand_worklist(ins: Instr) -> list[str]:
    """Operand field names whose current value is a GP register, in read-order."""
    out = []
    for f in _OPERAND_FIELDS:
        v = getattr(ins, f)
        if v is not None and _is_gp(v):
            out.append(f)
    return out


def _resolve_operand(dep: OperandDep, new_dest: dict[int, str]) -> Optional[str]:
    if dep.kind in (LOCAL, LOOP_INVARIANT, POST_LOOP):
        return new_dest.get(dep.producer_addr)
    if dep.kind == INTERLOOP:
        a = dep.interloop_bb0_addr if dep.interloop_bb0_addr is not None else dep.producer_addr
        return new_dest.get(a)
    return None  # UNDEFINED -- handled in phase 4


def _insert_bridges(
    da: DepAnalysis,
    sched: Schedule,
    addr_to: dict[int, Instr],
    new_dest: dict[int, str],
    branch: Instr,
) -> None:
    """Insert `mov bb0_reg, bb1_reg` for each distinct interloop pair."""
    # Collect distinct (BB0_producer, BB1_producer) pairs and their required cycle.
    pairs: dict[tuple[int, int], int] = {}
    for ins in da.bb1:
        for d in da.deps_of(ins.addr):
            if d.kind == INTERLOOP and d.interloop_bb0_addr is not None:
                p = addr_to[d.producer_addr]
                req = sched.cycle_of[p.addr] + latency_of(p.op)
                key = (d.interloop_bb0_addr, d.producer_addr)
                pairs[key] = max(pairs.get(key, 0), req)

    # Process pairs in ascending required-cycle order so easy movs slot in
    # before any push-down is forced.
    synth_addr = 1_000_000
    for (bb0_addr, bb1_addr), required in sorted(pairs.items(), key=lambda kv: kv[1]):
        mov = Instr(
            addr=synth_addr,
            op="mov",
            dest=new_dest[bb0_addr],
            src_a=new_dest[bb1_addr],
        )
        synth_addr += 1

        while True:
            bb1_end = sched.cycle_of[branch.addr]
            if bb1_end >= required:
                bundle = sched.bundles[bb1_end]
                if bundle[SLOT_ALU0] is None:
                    _place_mov(sched, mov, bb1_end, SLOT_ALU0)
                    break
                if bundle[SLOT_ALU1] is None:
                    _place_mov(sched, mov, bb1_end, SLOT_ALU1)
                    break
            _push_branch_down(sched, branch)


def _place_mov(sched: Schedule, mov: Instr, cycle: int, slot: int) -> None:
    sched.bundles[cycle][slot] = mov
    sched.cycle_of[mov.addr] = cycle
    sched.slot_of[mov.addr] = slot


# =========================================================================
# alloc_r: register allocation for the loop.pip schedule (handout §3.3.2)
# =========================================================================

def alloc_r(instrs: list[Instr], da: DepAnalysis, sched: PipSchedule) -> None:
    """Mutates `instrs` and `sched`. Requires sched.K to be set (true loop)."""
    if sched.K is None:
        return  # caller should fall back to alloc_b

    K = sched.K
    stride = K + 1
    new_dest: dict[int, int] = {}     # addr -> destination register index

    # phase 1 -- rotating regs for BB1 producers, scheduling order
    bb1_producers = [
        i for i in da.bb1
        if i.op not in BRANCH_OPS and i.writes() is not None and i.addr in sched.cycle_of
    ]
    bb1_producers.sort(key=lambda i: (sched.cycle_of[i.addr], sched.slot_of[i.addr]))
    rot = 32
    for ins in bb1_producers:
        new_dest[ins.addr] = rot
        rot += stride

    # phase 2 -- non-rotating regs for BB0 loop-invariant producers, in order
    # of first BB1 consumption
    nonrot = 1
    for c in da.bb1:
        for d in da.deps_of(c.addr):
            if d.kind == LOOP_INVARIANT and d.producer_addr not in new_dest:
                new_dest[d.producer_addr] = nonrot
                nonrot += 1

    # phase 4a -- BB0 producers of interloop values: dest = P.dest + 1 - St(P)
    for c in da.bb1:
        for d in da.deps_of(c.addr):
            if d.kind == INTERLOOP and d.interloop_bb0_addr is not None:
                bb0_addr = d.interloop_bb0_addr
                if bb0_addr in new_dest:
                    continue
                bb1_p = d.producer_addr
                if bb1_p not in new_dest:
                    continue
                p_idx = new_dest[bb1_p]
                p_stage = sched.stage_of.get(bb1_p, 0)
                new_dest[bb0_addr] = p_idx + 1 - p_stage

    # phase 4b -- remaining BB0/BB2 producers get fresh non-rotating regs
    for ins in list(da.bb0) + list(da.bb2):
        if ins.writes() is not None and ins.addr not in new_dest:
            new_dest[ins.addr] = nonrot
            nonrot += 1

    # apply destination renaming
    for ins in instrs:
        if ins.addr in new_dest:
            ins.dest = f"x{new_dest[ins.addr]}"

    # phase 3 -- rewrite operands per the formulas
    for ins in instrs:
        worklist = _operand_worklist(ins)
        for field, dep in zip(worklist, da.deps_of(ins.addr)):
            new_op = _resolve_pip(dep, ins, new_dest, sched, K)
            if new_op is not None:
                setattr(ins, field, new_op)

    # phase 4c -- undefined operand reads get fresh non-rotating regs
    sched_order = sorted(
        (i for i in instrs if i.addr in sched.cycle_of),
        key=lambda i: (sched.cycle_of[i.addr], sched.slot_of[i.addr]),
    )
    for ins in sched_order:
        worklist = _operand_worklist(ins)
        for field, dep in zip(worklist, da.deps_of(ins.addr)):
            if dep.kind == UNDEFINED:
                setattr(ins, field, f"x{nonrot}")
                nonrot += 1

    # phase 5 -- predicate every kernel BB1 instruction (not the branch)
    for ins in da.bb1:
        if ins.op in BRANCH_OPS:
            continue
        if ins.addr in sched.stage_of:
            ins.pred = f"p{32 + sched.stage_of[ins.addr]}"

    # phase 6 -- mov EC, K-1 ; mov p32, true in the bundle right before the
    # kernel; insert a fresh bundle if needed
    _insert_init_bundle(sched, K)


def _resolve_pip(
    dep: OperandDep,
    consumer: Instr,
    new_dest: dict[int, int],
    sched: PipSchedule,
    K: int,
) -> Optional[str]:
    if dep.kind == LOCAL:
        if dep.producer_addr not in new_dest:
            return None
        p_idx = new_dest[dep.producer_addr]
        if dep.producer_addr in sched.stage_of:
            c_stage = sched.stage_of.get(consumer.addr, 0)
            p_stage = sched.stage_of[dep.producer_addr]
            return f"x{p_idx + (c_stage - p_stage)}"
        return f"x{p_idx}"
    if dep.kind == LOOP_INVARIANT:
        if dep.producer_addr not in new_dest:
            return None
        return f"x{new_dest[dep.producer_addr]}"
    if dep.kind == INTERLOOP:
        bb1_p = dep.producer_addr
        if bb1_p not in new_dest:
            return None
        p_idx = new_dest[bb1_p]
        c_stage = sched.stage_of.get(consumer.addr, 0)
        p_stage = sched.stage_of.get(bb1_p, 0)
        return f"x{p_idx + (c_stage - p_stage) + 1}"
    if dep.kind == POST_LOOP:
        bb1_p = dep.producer_addr
        if bb1_p not in new_dest:
            return None
        p_idx = new_dest[bb1_p]
        p_stage = sched.stage_of.get(bb1_p, 0)
        return f"x{p_idx + (K - 1 - p_stage)}"
    return None  # UNDEFINED


def _insert_init_bundle(sched: PipSchedule, K: int) -> None:
    ec  = Instr(addr=2_000_001, op="mov", dest="EC",  imm=K - 1)
    p32 = Instr(addr=2_000_002, op="mov", dest="p32", bool_val=True)

    kernel_start = sched.kernel_start
    last_idx     = kernel_start - 1
    last_bundle  = sched.bundles[last_idx] if last_idx >= 0 else None

    free: list[int] = []
    if last_bundle is not None:
        if last_bundle[SLOT_ALU0] is None: free.append(SLOT_ALU0)
        if last_bundle[SLOT_ALU1] is None: free.append(SLOT_ALU1)

    if len(free) >= 2:
        last_bundle[free[0]] = ec
        last_bundle[free[1]] = p32
    elif len(free) == 1:
        last_bundle[free[0]] = ec
        new_b = [None] * 5
        new_b[SLOT_ALU0] = p32
        sched.bundles.insert(kernel_start, new_b)
        _shift_after_pip(sched, kernel_start)
    else:
        new_b = [None] * 5
        new_b[SLOT_ALU0] = ec
        new_b[SLOT_ALU1] = p32
        sched.bundles.insert(kernel_start, new_b)
        _shift_after_pip(sched, kernel_start)


def _shift_after_pip(sched: PipSchedule, insert_idx: int) -> None:
    """One bundle was inserted at insert_idx; shift everything at >= idx."""
    for addr, c in list(sched.cycle_of.items()):
        if c >= insert_idx:
            sched.cycle_of[addr] = c + 1
    if sched.kernel_start is not None and sched.kernel_start >= insert_idx:
        sched.kernel_start += 1
    if sched.bb2_start is not None and sched.bb2_start >= insert_idx:
        sched.bb2_start += 1
    if sched.branch is not None and sched.branch.target is not None and sched.branch.target >= insert_idx:
        sched.branch.target += 1


# =========================================================================

def _push_branch_down(sched: Schedule, branch: Instr) -> None:
    """Move the loop branch one bundle later, shifting BB2 cycles up by one."""
    cur = sched.cycle_of[branch.addr]
    sched.bundles[cur][SLOT_BRANCH] = None

    new_cycle = cur + 1
    sched.bundles.insert(new_cycle, [None] * 5)
    sched.bundles[new_cycle][SLOT_BRANCH] = branch

    for addr, c in list(sched.cycle_of.items()):
        if addr == branch.addr:
            continue
        if c >= new_cycle:
            sched.cycle_of[addr] = c + 1

    sched.cycle_of[branch.addr] = new_cycle
    sched.bb1_end = new_cycle
    if sched.bb2_start is not None:
        sched.bb2_start = new_cycle + 1
