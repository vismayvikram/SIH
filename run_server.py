"""
SIH26012 Feature Review Platform - Local Development Server
Launches the FastAPI backend and serves the frontend web-GIS interface at http://127.0.0.1:8000.
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 70)
    print(" SIH26012 Geospatial Feature Review Platform")
    print(f" Starting local server at: http://{host}:{port}/")
    print(" Access the Web UI in your browser at the URL above.")
    print("=" * 70)
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)
