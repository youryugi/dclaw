#!/usr/bin/env bash
# Source this file before compiling or running the validation application:
#   source validation/usgs_flume_2017/environment.sh

export CLAW="${CLAW:-/home/yang/github/clawpack-runtime}"
if [[ ":${PYTHONPATH:-}:" != *":${CLAW}:"* ]]; then
    export PYTHONPATH="${CLAW}${PYTHONPATH:+:${PYTHONPATH}}"
fi

