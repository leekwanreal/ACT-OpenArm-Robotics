#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convenience entry point for Dataset Resampling & Synchronization
Usage:
    python resample_dataset.py --input_dir data_set/data_set --output_dir data_set/data_set_50hz
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from act_pipeline.data.resample_dataset import main

if __name__ == "__main__":
    main()

