#!/bin/bash
# usage: run.sh <input.json> <simple.json> <pip.json>
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 -m scheduler.main "$1" "$2" "$3"
