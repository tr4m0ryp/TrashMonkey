"""Module entry point: ``python -m trashmonkey.smoke``."""

import sys

from trashmonkey.smoke.cli import main

if __name__ == "__main__":
    sys.exit(main())
