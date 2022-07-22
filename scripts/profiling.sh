#!/bin/bash
nsys profile -t cuda,nvtx,cublas,cudnn -f true -o profiling python main.py
