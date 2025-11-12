# ============================================================================
# Face Model - Direct Database Comparison
# ============================================================================

import cv2
import numpy as np
import threading
from insightface.app import FaceAnalysis
from utils import load_posts, cosine_similarity

_face_app = None
_face_app_lock = threading.Lock()


def get_face_app():
    global _face_app
    with _face_app_lock:
        if _face_app is not None:
            return _face_app

        try:
            print("[INFO] Preparing face model with CUDA...")
            _face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            _face_app.prepare(ctx_id=0)
            print("[INFO] Model ready (CUDA).")
            return _face_app
        except Exception as e_cuda:
            print(f"[WARN] CUDA failed: {e_cuda}. Trying CPU...")

        try:
            _face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            _face_app.prepare(ctx_id=0)
            print("[INFO] Model ready (CPU).")
            return _face_app
        except Exception as e_cpu:
            print(f"[ERROR] Failed to prepare face model: {e_cpu}")
            _face_app = None
            return None


def get_face_embedding_from_image(image):
    app = get_face_app()
    if app is None:
        return None

    try:
        faces = app.get(image)
        if len(faces) == 0:
            return None
        return np.array(faces[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None


def compare_embedding_with_posts(embedding: np.ndarray) -> float:

    try:
        posts = load_posts()
        if not posts:
            return 0.0

        max_similarity = 0.0
        for post in posts:
            try:
                post_emb = np.array(post['embedding'], dtype=np.float32)
                sim = cosine_similarity(embedding, post_emb)
                if sim > max_similarity:
                    max_similarity = sim
            except Exception:
                continue

        return float(max_similarity)
    except Exception as e:
        print(f"[ERROR] Database comparison failed: {e}")
        return 0.0


def images_to_embedding_list(image_paths, index_manager=None):

    from config import SIMILARITY_THRESHOLD

    embeddings = []
    for i, p in enumerate(image_paths):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read image: {p}")
            continue

        emb = get_face_embedding_from_image(img)
        if emb is not None:
            embeddings.append(emb)
            print(f"[INFO] Extracted embedding {i+1}/{len(image_paths)}")

    if not embeddings:
        print("[ERROR] No embeddings extracted!")
        return None

    if len(embeddings) == 1:
        print("[INFO] Single image, using directly")
        return embeddings[0]

    try:
        posts = load_posts()
        if posts and len(posts) > 0:
            print(f"[INFO] Comparing {len(embeddings)} images with {len(posts)} posts...")

            candidates = []
            for i, emb in enumerate(embeddings):
                max_sim = compare_embedding_with_posts(emb)
                candidates.append((i, emb, max_sim))
                print(f"  Image {i+1}: max database similarity = {max_sim:.4f}")

            candidates.sort(key=lambda x: x[2], reverse=True)

            best_idx, best_emb, best_sim = candidates[0]

            if best_sim >= SIMILARITY_THRESHOLD:
                print(f"[INFO] Selected image {best_idx+1} with similarity {best_sim:.4f}")
                return best_emb
            else:
                print(f"[INFO] All images < {SIMILARITY_THRESHOLD}, using mean of all")
                return np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        else:
            print("[INFO] No posts in database, using mean of all embeddings")
            return np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    except Exception as e:
        print(f"[ERROR] Failed to process embeddings: {e}")
        import traceback
        traceback.print_exc()
        print("[INFO] Falling back to simple mean")
        return np.mean(np.stack(embeddings), axis=0).astype(np.float32)
