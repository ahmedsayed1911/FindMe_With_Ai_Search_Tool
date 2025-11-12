# ============================================================================
# Utility Functions
# ============================================================================

import json
import numpy as np
from config import POSTS_JSON


def load_posts():
    if not POSTS_JSON.exists():
        return []
    try:
        with open(POSTS_JSON, 'r') as f:
            return json.load(f)
    except:
        return []


def save_posts(posts):
    with open(POSTS_JSON, 'w') as f:
        json.dump(posts, f, indent=2)


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
