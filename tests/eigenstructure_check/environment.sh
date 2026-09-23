#!/usr/bin/env bash
# Source before compiling or running:
#   source tests/dev/eigenstructure_check/environment.sh

export CLAW="${CLAW:-/home/yang/github/clawpack-runtime}"
if [[ ":${PYTHONPATH:-}:" != *":${CLAW}:"* ]]; then
    export PYTHONPATH="${CLAW}${PYTHONPATH:+:${PYTHONPATH}}"
fi
