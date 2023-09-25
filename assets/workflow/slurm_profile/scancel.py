#!/usr/bin/env python3

import os
import subprocess as sp
import shlex
import logging
import sys
from collections import defaultdict

logger = logging.getLogger(__name__)

args = sys.argv[1:]

jobs = defaultdict(list)

for arg in args:
    jobid, cluster = arg.split(";")
    jobs[cluster].append(jobid)


for (cluster, jobids) in jobs.items():
    jobids = " ".join(jobids)
    sacct_res = sp.check_output(shlex.split(f"scancel --cluster={cluster} {jobid}"))