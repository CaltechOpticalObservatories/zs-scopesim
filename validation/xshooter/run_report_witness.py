"""Run one supplied Fremling report witness at a fixed observation epoch.

The report folder is a user-provided input, not a dependency of the simulator.
Select ScopeSim with PYTHONPATH and the instrument checkout with IRDB.
Only the unspecified observation date is changed; physical effects stay enabled.
"""
import argparse
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('demo', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.demo.resolve().parent))
import _common

original_train = _common.train


def dated_train(slit=.70, **properties):
    properties['!OBS.mjdobs'] = 60310.25
    return original_train(slit, **properties)


_common.train = dated_train
runpy.run_path(str(args.demo), run_name='__main__')
