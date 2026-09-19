"""CLI entrypoint: `python -m utils.storage [migrate|version]`."""
import sys

from utils.storage.migrations import _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
