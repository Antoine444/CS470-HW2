# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

Implement a VLIW scheduler for the CS-470 Advanced Computer Architecture homework. The scheduler reads a flat list of instructions (JSON) and outputs two VLIW schedules:

- **simple.json** — loop scheduling using the `loop` instruction
- **pip.json** — software-pipelined scheduling using the `loop.pip` instruction

You must provide three scripts: `build.sh`, `run.sh`, and your scheduler implementation. The grading environment is the provided Docker image (Ubuntu 22.04 with Python 3, Go 1.20, Rust, Scala/sbt, Node 18, Java 17).

## Running and Testing

```bash
# Build your solution
./build.sh

# Run on all test cases (produces simple.json and pip.json per test)
./runall.sh   # calls ./run.sh <input.json> <simple.json> <pip.json> for each test

# Compare outputs against reference solutions
./testall.sh  # prints PASSED/FAILED per test for both loop and loop.pip schedules

# Run and test a single test case manually
./run.sh given_tests/03/input.json given_tests/03/simple.json given_tests/03/pip.json
python compare.py --loop given_tests/03/simple.json --refLoop given_tests/03/simple_ref.json
python compare.py --pip given_tests/03/pip.json --refPip given_tests/03/pip_ref.json

# Simulate a schedule (optional, for debugging)
python simulator/vliw470.py simulator/program.json /tmp/result.json --memory simulator/memory.json
```

`run.sh` receives exactly 3 arguments: input path, output path for simple schedule, output path for pip schedule.

## VLIW470 Architecture

Each **bundle** (VLIW instruction word) has exactly 5 slots in this order:

| Index | Unit    | Instructions              |
|-------|---------|---------------------------|
| 0     | ALU0    | add, addi, sub, mov, nop  |
| 1     | ALU1    | add, addi, sub, mov, nop  |
| 2     | Mult    | mulu, nop (2-cycle latency)|
| 3     | Mem     | ld, st, nop (1-cycle latency) |
| 4     | Branch  | loop, loop.pip, nop       |

### Registers

- **x0–x31**: general-purpose (x0 is always 0 in practice)
- **x32–x63**: rotating registers — physical index = logical index − RBB (mod 32) + 64 if needed
- **p0–p95**: predicate registers; instructions can be predicated `(pN) inst ...`
- **LC**: loop count, **EC**: epilogue count, **RBB**: rotating base register (0–63)

### Loop instructions

- `loop <target_bundle>` — if LC > 0: LC−−, jump to target
- `loop.pip <target_bundle>` — pipelined loop:
  - If LC > 0: LC−−, RBB++, p(rename(32))=true, jump to target (prologue/kernel stage)
  - Elif EC > 0: EC−−, RBB++, p(rename(32))=false, jump to target (epilogue stage)
  - Else: p(rename(32))=false, fall through (done)

### Instruction format

Input JSON is a flat array of instruction strings, e.g.:
```json
["mov LC, 100", "addi x9, x0, 10", "ld x1, 0(x2)", "add x3, x4, x5", "loop 2"]
```

Output JSON is an array of bundles (arrays of 5 strings). Empty slots must be `"nop"`. The comparator allows ALU0 and ALU1 to be swapped.

## Data Hazards

- ALU results available next cycle (1-cycle latency)
- `mulu` result available 2 cycles later (2-cycle latency)
- `ld` result available next cycle (1-cycle latency)
- The `loop`/`loop.pip` branch instruction in a bundle executes in the same cycle as the other instructions in that bundle; the branch target is the first bundle of the loop body

## Reference Tests

`given_tests/` contains numbered test directories (02–17 and more). Each has:
- `input.json` — flat instruction list
- `desc.txt` — human-readable description of the test scenario
- `simple_ref.json` / `pip_ref.json` — reference outputs (there may be multiple valid solutions)
