# CS-470 Homework 2 — VLIW470 Scheduler

## Build & run

```bash
./build.sh                                          # no-op (pure Python)
./run.sh <input.json> <simple.json> <pip.json>      # produces both schedules
```

Python 3 (the grading container ships 3.10) is the only dependency.

## Implementation

Pure Python under `scheduler/`:

| File | Role |
|---|---|
| `parse.py` | Tokenize instructions, normalize immediates to decimal, split BB0/BB1/BB2 |
| `deps.py` | True-dependency analysis: local / interloop / loop_invariant / post_loop / undefined |
| `schedule_loop.py` | ASAP scheduler for `loop`; pushes the branch down to enlarge II if interloop deps would be violated |
| `schedule_pip.py` | Modulo scheduler for `loop.pip`; iterates II from `II_res = max(ceil(N_alu/2), N_mult, N_mem, 1)` upward |
| `alloc.py` | `alloc_b` (simple loop): rename + bridging `mov`s; `alloc_r` (pip): rotating registers (stride K+1), predicate insertion, EC/p32 init bundle |
| `emit.py` | Bundle -> JSON serialization |
| `main.py` | CLI entry point |

The handout's ASAP / modulo-scheduling algorithms (Sections 3.2 and 3.3) are followed directly. All 17 provided reference tests pass for both `simple.json` and `pip.json`.
