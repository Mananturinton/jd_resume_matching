"""
watch_dependency_trees.py — watches combined_classification.json and reruns
generate_dependency_trees.py every time it changes (polling on mtime; no
fswatch/watchdog dependency required).

Usage: python3 watch_dependency_trees.py
Stop with Ctrl-C (or kill the background process).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "combined_classification.json")
GENERATOR = os.path.join(HERE, "generate_dependency_trees.py")
POLL_SECONDS = 1.0


def run_once():
    proc = subprocess.run([sys.executable, GENERATOR], capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        print(f"[watch] generate_dependency_trees.py failed:\n{proc.stderr}", file=sys.stderr)
    else:
        print("[watch] dependency trees regenerated")


def main():
    print(f"[watch] watching {TARGET} for changes (polling every {POLL_SECONDS}s)")
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(TARGET)
        except FileNotFoundError:
            time.sleep(POLL_SECONDS)
            continue
        if mtime != last_mtime:
            if last_mtime is not None:
                print(f"[watch] {os.path.basename(TARGET)} changed, regenerating...")
            run_once()
            last_mtime = mtime
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
