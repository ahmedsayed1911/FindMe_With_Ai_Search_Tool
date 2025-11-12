# ============================================================================
# ChromaDB Manager
# ============================================================================

import chromadb
import numpy as np
from pathlib import Path
import sys
from config import SIMILARITY_THRESHOLD
from utils import load_posts

class ChromaManager:
    def __init__(self, persist_directory=None):
        if persist_directory is None:
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent
            
            persist_directory = str(base_dir / "chroma_db")
        
        print(f"[INFO] ChromaDB storage path: {persist_directory}")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        if self.get_count() == 0:
            self.rebuild_from_posts()
    
    def rebuild_from_posts(self):
        """Rebuild ChromaDB from existing posts.json"""
        try:
            posts = load_posts()
            if not posts:
                print("[INFO] No posts to add to ChromaDB")
                return
            
            print(f"[INFO] Rebuilding ChromaDB with {len(posts)} posts...")
            
            ids = []
            embeddings = []
            metadatas = []
            
            for post in posts:
                post_id = post.get('post_id')
                if not post_id:
                    continue
                
                try:
                    embedding = post.get('embedding')
                    if not embedding:
                        continue
                    
                    ids.append(f"post_{post_id}")
                    embeddings.append(embedding)
                    metadatas.append({
                        'num_images': len(post.get('images', [])),
                        'post_id': post_id
                    })
                    
                except Exception as e:
                    print(f"[WARN] Failed to process post {post_id}: {e}")
                    continue
            
            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"[INFO] Successfully added {len(ids)} posts to ChromaDB")
            else:
                print("[WARN] No valid posts found to add to ChromaDB")
                
        except Exception as e:
            print(f"[ERROR] Failed to rebuild ChromaDB: {e}")
    
    def add_post(self, post_id: int, embedding: np.ndarray, metadata: dict = None):
        """Add post embedding to ChromaDB"""
        try:
            if metadata is None:
                metadata = {}
            
            metadata['post_id'] = post_id
            
            self.collection.add(
                ids=[f"post_{post_id}"],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
            print(f"[INFO] Added post {post_id} to ChromaDB")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add post {post_id} to ChromaDB: {e}")
            return False
    
    def delete_post(self, post_id: int):
        """Delete post from ChromaDB"""
        try:
            self.collection.delete(ids=[f"post_{post_id}"])
            print(f"[INFO] Deleted post {post_id} from ChromaDB")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete post {post_id} from ChromaDB: {e}")
            return False
    
    def query_similar(self, query_embedding: np.ndarray, n_results: int = 100):
        """Query similar posts using cosine similarity"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["distances", "metadatas"]
            )
            return results
        except Exception as e:
            print(f"[ERROR] ChromaDB query failed: {e}")
            return None
    
    def get_count(self) -> int:
        """Get total number of posts in ChromaDB"""
        try:
            return self.collection.count()
        except:
            return 0
    
    def get_all_ids(self):
        """Get all post IDs in ChromaDB for debugging"""
        try:
            results = self.collection.get()
            return results['ids'] if results and 'ids' in results else []
        except:
            return []
    
    def verify_embeddings(self):
        """Verify that ChromaDB embeddings match JSON embeddings"""
        try:
            posts = load_posts()
            chroma_data = self.collection.get()
            
            print(f"[VERIFY] JSON posts: {len(posts)}, ChromaDB posts: {len(chroma_data['ids'])}")
            
            for i, post_id_str in enumerate(chroma_data['ids']):
                post_id = int(post_id_str.split('_')[1])
                chroma_emb = chroma_data['embeddings'][i]
                
                # Find corresponding post in JSON
                json_post = next((p for p in posts if p['post_id'] == post_id), None)
                if json_post:
                    json_emb = json_post['embedding']
                    
                    # Calculate similarity between the two embeddings
                    from utils import cosine_similarity
                    similarity = cosine_similarity(np.array(json_emb), np.array(chroma_emb))
                    print(f"[VERIFY] Post {post_id}: similarity between JSON and ChromaDB = {similarity:.4f}")
                    
                    if similarity < 0.99:
                        print(f"[WARNING] Post {post_id} has different embeddings in JSON and ChromaDB!")
                else:
                    print(f"[WARNING] Post {post_id} not found in JSON")
                    
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
    
    def force_rebuild(self):
        """Force rebuild ChromaDB from scratch"""
        try:
            self.client.delete_collection("face_embeddings")
            
            self.collection = self.client.get_or_create_collection(
                name="face_embeddings", 
                metadata={"hnsw:space": "cosine"}
            )
            
            self.rebuild_from_posts()
            print("[INFO] ChromaDB force rebuild completed")
            
        except Exception as e:
            print(f"[ERROR] Force rebuild failed: {e}")

if __name__ == "__main__":
    print("Testing ChromaManager...")
    cm = ChromaManager()
    print(f"ChromaDB has {cm.get_count()} posts")