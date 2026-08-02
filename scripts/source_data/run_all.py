"""Run every build_fig*.py script in order, then assemble the workbook.

Usage: python scripts/source_data/run_all.py
(build_fig6.py is slow - 20-40+ min - see README.md in this folder)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "build_fig1.py",
    "build_fig2.py",
    "build_fig3_fig4.py",
    "build_fig5.py",
    "build_fig6.py",
    "build_fig7.py",
    "assemble_workbook.py",
]

for script in SCRIPTS:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(HERE / script)], check=True)
