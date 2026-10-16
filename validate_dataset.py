#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Verification Script for OpenArm ACT (16-DOF + RGB-D)
Repo: OpenArm_ACT_Training
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from act_pipeline.data.validate_dataset import main

if __name__ == "__main__":
    main()

