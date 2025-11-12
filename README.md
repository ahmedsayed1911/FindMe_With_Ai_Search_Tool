README.md
Missing Persons Finder – Admin Panel

This project provides an intelligent admin system for managing and identifying missing persons using face recognition and vector similarity search. It combines InsightFace, ChromaDB, and a PyQt5 GUI to enable fast and accurate matching of facial embeddings.

The system allows admins to:

Add new missing person posts

Upload images and extract face embeddings

Search for similar faces in the database

View and manage all stored posts

Automatically rebuild the vector database when needed

🚀 Key Features
✔ Face Embedding Extraction

Uses InsightFace (Buffalo_L) model

Supports GPU (CUDA) automatically with fallback to CPU

Extracts a 512-dimensional vector for every face

✔ Fast Vector Search with ChromaDB

Stores embeddings in a persistent vector database

Uses HNSW indexing with cosine similarity

Supports add, delete, and similarity search operations

✔ Admin Dashboard (PyQt5)

Search UI

Results UI

Feed of posts

Add post widget

Auto-refresh for updated data

✔ Automatic Database Sync

All posts are stored in posts.json

ChromaDB is rebuilt automatically if empty
