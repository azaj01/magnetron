#!/usr/bin/env bash

pyinstrument --show-all -r html -o profile.html examples/qwen3/main.py --prompt "How big is the moon short, short answer?"
