#!/usr/bin/env bash
# Source this file before compiling or running the validation application:
#   source validation/montecito_realterrain_crosscheck/environment.sh

export CLAW="${CLAW:-/home/yang/github/clawpack-runtime}"
if [[ ":${PYTHONPATH:-}:" != *":${CLAW}:"* ]]; then
    export PYTHONPATH="${CLAW}${PYTHONPATH:+:${PYTHONPATH}}"
fi
