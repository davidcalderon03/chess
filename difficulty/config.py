"""Adjustable settings for the difficulty module. Every value can be overridden by an
environment variable or a command-line flag of the scripts.

Importing this module also puts the repo root on sys.path, so `import extract`, `import dataset`
and `import train` work (we only ever import from them, never edit them).
"""
import os
import shutil
import sys
from pathlib import Path

DIFF_DIR = Path(__file__).resolve().parent
ROOT = DIFF_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Stockfish -------------------------------------------------------------------------
# Location: $STOCKFISH_PATH, else `stockfish` on PATH, else difficulty/bin/stockfish.
STOCKFISH_PATH = (os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
                  or str(DIFF_DIR / "bin" / "stockfish"))
# Fixed search budget = number of nodes per position. With 1 thread and a fresh game state
# for every position this is repeatable (unlike a time limit, which depends on machine load).
STOCKFISH_NODES = int(os.environ.get("STOCKFISH_NODES", 200_000))
STOCKFISH_HASH_MB = 64
STOCKFISH_THREADS = 1

# --- Data / model ------------------------------------------------------------------------
# Data: $CHESS_DATA, else the team's repo-root data/ if it exists, else our local cache.
DATA_DIR = os.environ.get("CHESS_DATA") or str(
    ROOT / "data" if (ROOT / "data").exists() else DIFF_DIR / "data_cache" / "data")
MODEL_PATH = os.environ.get("CHESS_MODEL", str(ROOT / "model.pt"))
RESULTS_DIR = DIFF_DIR / "results"
STOCKFISH_CSV = RESULTS_DIR / "stockfish_test.csv"
DIFFICULTY_CSV = RESULTS_DIR / "difficulty.csv"
RATING_SUPPORT_JSON = RESULTS_DIR / "rating_support.json"

# --- Rating grid -------------------------------------------------------------------------
GRID_STEP = 100
FALLBACK_GRID = (1200, 2400)     # used ONLY if rating_support.json does not exist (with a warning)
MIN_COUNT_PER_BIN = 5_000        # a rating bin needs at least this many training positions
BIN_HALF_WIDTH = 100             # a position counts for rating R if both players are within +-100

# --- Inputs a future model may need that today's data cannot provide --------------------------
# Used whenever a model asks for one of these and the caller does not supply it.
# Every use is recorded in the output ("defaults_used").
DEFAULT_CONTEXT = {
    "white_clock_s": 600.0,      # both players with plenty of time left
    "black_clock_s": 600.0,
    "time_control": "600+0",     # base seconds + increment seconds
    "history": None,             # recent game history (list of moves); None = not available
}
