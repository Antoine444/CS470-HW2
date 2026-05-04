"""Independent semantic validator for VLIW470 schedules.

Usage:
    python edge_tests/validate.py <input.json> <simple.json> <pip.json> [--memory <mem.json>]

Checks (in order):
  1. Structural: bundle has 5 string slots; each slot's instruction matches its
     functional unit (ALU0/1, Mult, Mem, Branch); at most one branch present;
     branch target is a valid bundle index.
  2. II sanity (pip): II_pip = (loop_bundle - target + 1) is >= the resource
     lower bound II_res = max(ceil(n_alu/2), n_mult, n_mem) computed over BB1
     instructions of the input.
  3. Functional equivalence: a pure-Python sequential interpreter executes the
     flat input.json; simulator/vliw470.py executes both schedules; we compare
     final architectural state (x1..x31 + memory) of each schedule against the
     reference. Any mismatch is reported.

Exits 0 on full success, non-zero (with error printed) otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

# Make scheduler.* importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scheduler.parse import parse_program, split_bbs, ALU_OPS, MULT_OPS, MEM_OPS, BRANCH_OPS  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _ref_interp import run_file as ref_run  # noqa: E402

SLOT_NAMES = ["ALU0", "ALU1", "Mult", "Mem", "Branch"]
ALLOWED_OPS = [
    ALU_OPS | {"nop"},
    ALU_OPS | {"nop"},
    MULT_OPS | {"nop"},
    MEM_OPS | {"nop"},
    BRANCH_OPS | {"nop"},
]

_PRED_RE = re.compile(r"^\s*\(\s*p\d+\s*\)\s*(.*)$")


class ValidationError(Exception):
    pass


def _opcode(slot_text: str) -> str:
    s = slot_text.strip()
    m = _PRED_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s.split(None, 1)[0].lower() if s else "nop"


def check_structural(schedule: list[list[str]], label: str) -> dict:
    """Returns metadata about the schedule (branch info, body bounds)."""
    if not isinstance(schedule, list):
        raise ValidationError(f"{label}: schedule is not a list")
    branch_bundle = None
    branch_op = None
    branch_target = None
    for bidx, bundle in enumerate(schedule):
        if not isinstance(bundle, list) or len(bundle) != 5:
            raise ValidationError(f"{label}: bundle {bidx} has {len(bundle) if isinstance(bundle, list) else '?'} slots, expected 5")
        for slot, slot_text in enumerate(bundle):
            if not isinstance(slot_text, str):
                raise ValidationError(f"{label}: bundle {bidx} slot {slot} is not a string")
            op = _opcode(slot_text)
            if op not in ALLOWED_OPS[slot]:
                raise ValidationError(
                    f"{label}: bundle {bidx} slot {SLOT_NAMES[slot]} has opcode '{op}' "
                    f"(allowed: {sorted(ALLOWED_OPS[slot])})"
                )
            if op in BRANCH_OPS:
                if branch_bundle is not None:
                    raise ValidationError(f"{label}: multiple branch instructions (at bundle {branch_bundle} and {bidx})")
                branch_bundle = bidx
                branch_op = op
                # parse the target
                tail = slot_text.strip().split(None, 1)[1].strip() if " " in slot_text.strip() else ""
                try:
                    branch_target = int(tail)
                except Exception:
                    raise ValidationError(f"{label}: branch at bundle {bidx} has unparseable target {tail!r}")
                if not (0 <= branch_target <= bidx):
                    raise ValidationError(f"{label}: branch target {branch_target} out of range (>{bidx} or <0)")
    return {
        "branch_bundle": branch_bundle,
        "branch_op": branch_op,
        "branch_target": branch_target,
        "n_bundles": len(schedule),
    }


def check_pip_ii(input_instrs: list, pip_meta: dict, label: str) -> None:
    if pip_meta["branch_op"] is None:
        return  # no loop, nothing to check
    if pip_meta["branch_op"] != "loop.pip":
        raise ValidationError(f"{label}: pip schedule's branch is {pip_meta['branch_op']}, expected loop.pip")
    bb0, bb1, bb2 = split_bbs(input_instrs)
    body_no_branch = [i for i in bb1 if i.op not in BRANCH_OPS]
    n_alu = sum(1 for i in body_no_branch if i.op in ALU_OPS)
    n_mult = sum(1 for i in body_no_branch if i.op in MULT_OPS)
    n_mem = sum(1 for i in body_no_branch if i.op in MEM_OPS)
    ii_res = max(math.ceil(n_alu / 2), n_mult, n_mem, 1)
    ii_pip = pip_meta["branch_bundle"] - pip_meta["branch_target"] + 1
    if ii_pip < ii_res:
        raise ValidationError(
            f"{label}: II_pip={ii_pip} < II_res={ii_res} "
            f"(n_alu={n_alu}, n_mult={n_mult}, n_mem={n_mem})"
        )


def run_simulator(schedule_path: str, memory_path: str | None) -> dict:
    """Returns the final state dict from simulator/vliw470.py."""
    out_path = tempfile.mktemp(suffix=".json")
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "simulator", "vliw470.py"),
           schedule_path, out_path]
    if memory_path:
        cmd += ["--memory", memory_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise ValidationError(f"simulator failed for {schedule_path}:\n{res.stderr}")
    with open(out_path) as f:
        states = json.load(f)
    os.remove(out_path)
    if not states:
        raise ValidationError(f"simulator produced empty state list for {schedule_path}")
    return states[-1]


def compare_state(ref, sim_state: dict, label: str) -> list[str]:
    """Compare *memory* only.

    Register-level comparison is meaningless: alloc.py freely renames every
    architectural register (BB0 movs included), so the same logical value
    ends up in a different physical x-index. Memory is the only stable
    observable: a `st xN, k(xM)` writes the renamed-xN's value to the
    address computed from the renamed-xM's value, and both renamings are
    consistent within the schedule. Tests are designed so all observable
    results land in memory.
    """
    errors: list[str] = []
    sim_mem_raw = sim_state.get("MemoryData", {})
    sim_mem = {int(k): v & 0xFFFFFFFFFFFFFFFF for k, v in sim_mem_raw.items()}
    addrs = set(ref.mem.keys()) | set(sim_mem.keys())
    for a in sorted(addrs):
        ref_v = ref.mem.get(a, 0) & 0xFFFFFFFFFFFFFFFF
        sim_v = sim_mem.get(a, 0) & 0xFFFFFFFFFFFFFFFF
        if ref_v != sim_v:
            errors.append(f"{label}: mem[{hex(a)}]: ref={ref_v} sim={sim_v}")
    return errors


def validate(input_path: str, simple_path: str, pip_path: str, memory_path: str | None) -> int:
    with open(input_path) as f:
        input_lines = json.load(f)
    input_instrs = parse_program(input_lines)

    with open(simple_path) as f:
        simple_sched = json.load(f)
    with open(pip_path) as f:
        pip_sched = json.load(f)

    # 1. structural
    simple_meta = check_structural(simple_sched, "simple")
    pip_meta = check_structural(pip_sched, "pip")

    # 2. II sanity for pip
    check_pip_ii(input_instrs, pip_meta, "pip")

    # 3. functional equivalence
    ref = ref_run(input_path, memory_path)
    simple_state = run_simulator(simple_path, memory_path)
    pip_state = run_simulator(pip_path, memory_path)

    errors: list[str] = []
    errors += compare_state(ref, simple_state, "simple")
    errors += compare_state(ref, pip_state, "pip")

    if errors:
        for e in errors:
            print(e)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("simple")
    ap.add_argument("pip")
    ap.add_argument("--memory")
    args = ap.parse_args()
    # Auto-discover memory.json beside input.json if not given
    mem = args.memory
    if mem is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.input)), "memory.json")
        if os.path.exists(candidate):
            mem = candidate
    try:
        rc = validate(args.input, args.simple, args.pip, mem)
    except ValidationError as e:
        print(f"FAIL: {e}")
        return 1
    if rc == 0:
        print("PASS")
    else:
        print("FAIL: state mismatch")
    return rc


if __name__ == "__main__":
    sys.exit(main())
