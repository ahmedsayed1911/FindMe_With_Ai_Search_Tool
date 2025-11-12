# ============================================================================
# Configuration
# ============================================================================

from pathlib import Path
import sys
import os

# تحديد إذا كان التطبيق شغال كـ EXE أو كـ Python
if getattr(sys, 'frozen', False):
    # إذا شغال كـ EXE
    BASE_DIR = Path(sys.executable).parent
else:
    # إذا شغال كـ Python
    BASE_DIR = Path(__file__).parent

# Directories
KNOWN_DIR = BASE_DIR / "posts"
KNOWN_DIR.mkdir(parents=True, exist_ok=True)

POSTS_JSON = BASE_DIR / "posts.json"

# Thresholds
SIMILARITY_THRESHOLD = 0.20
OUTLIER_HIGH_THRESHOLD = 0.85
OUTLIER_LOW_THRESHOLD = 0.25

# Settings
MAX_IMAGES = 5
AUTO_REFRESH_MS = 3000

print(f"[CONFIG] Running as EXE: {getattr(sys, 'frozen', False)}")
print(f"[CONFIG] Base directory: {BASE_DIR}")
print(f"[CONFIG] SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
print(f"[CONFIG] AUTO_REFRESH_MS = {AUTO_REFRESH_MS}")