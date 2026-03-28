"""
Standalone Web Application - Game Development AI Assistant

A FastAPI-based web server providing a REST API and web UI for the
multimodal game development AI assistant.

Run with: uvicorn app:app --reload
Access at: http://localhost:8000
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import base64
from datetime import datetime

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart")

# Import our multimodal assistant
try:
    from multimodal_assistant import MultimodalGameDevAssistant, MultimodalDocumentProcessor
    ASSISTANT_AVAILABLE = True
except ImportError:
    ASSISTANT_AVAILABLE = False
    print("Warning: multimodal_assistant not available")

if not FASTAPI_AVAILABLE:
    raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn python-multipart")

# Initialize FastAPI app
app = FastAPI(
    title="Game Dev AI Assistant",
    description="Multimodal AI assistant for game development teams",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant: Optional[MultimodalGameDevAssistant] = None
VECTOR_STORE_PATH = "vector_store_web"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models for API
class QueryRequest(BaseModel):
    """クエリリクエスト."""
    query: str
    k: int = 5
    modality: Optional[str] = None  # "text", "image", or None


class QueryResponse(BaseModel):
    """クエリレスポンス."""
    query: str
    results: List[Dict[str, Any]]
    timestamp: str


class StatusResponse(BaseModel):
    """ステータスレスポンス."""
    status: str
    vector_store_loaded: bool
    num_items: int
    message: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化."""
    global assistant

    print("Starting Game Dev AI Assistant...")

    # Load or create assistant
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Loading existing vector store from {VECTOR_STORE_PATH}")
        assistant = MultimodalGameDevAssistant(vector_store_path=VECTOR_STORE_PATH)
        print(f"Loaded {len(assistant.vector_store.items)} items")
    else:
        print("No vector store found. Creating new assistant.")
        assistant = MultimodalGameDevAssistant()
        print("Assistant ready. Add documents via /api/upload or /api/build")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """メインページ."""
    return get_html_ui()


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """アシスタントのステータスを取得."""
    if assistant is None:
        return StatusResponse(
            status="not_initialized",
            vector_store_loaded=False,
            num_items=0,
            message="Assistant not initialized"
        )

    return StatusResponse(
        status="ready",
        vector_store_loaded=True,
        num_items=len(assistant.vector_store.items),
        message=f"Ready with {len(assistant.vector_store.items)} items"
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """テキストクエリで検索."""
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    try:
        response = assistant.ask(
            query=request.query,
            k=request.k,
            modality=request.modality
        )

        # Convert results to JSON-serializable format
        results = []
        for result in response['results']:
            item = result['item'].copy()

            # Convert image to base64 for display
            if item['type'] == 'image' and os.path.exists(item['path']):
                try:
                    with open(item['path'], 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()
                        item['image_data'] = f"data:image/png;base64,{image_data}"
                except Exception as e:
                    print(f"Error loading image {item['path']}: {e}")
                    item['image_data'] = None

            results.append({
                "item": item,
                "score": result['score']
            })

        return QueryResponse(
            query=request.query,
            results=results,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query-image")
async def query_image(
    file: UploadFile = File(...),
    k: int = Form(5),
    modality: Optional[str] = Form(None)
):
    """画像クエリで検索."""
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    # Save uploaded image
    image_path = UPLOAD_DIR / f"query_{datetime.now().timestamp()}_{file.filename}"

    try:
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Query with image
        response = assistant.ask(
            query="",
            query_image=str(image_path),
            k=k,
            modality=modality
        )

        # Convert results
        results = []
        for result in response['results']:
            item = result['item'].copy()

            if item['type'] == 'image' and os.path.exists(item['path']):
                try:
                    with open(item['path'], 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()
                        item['image_data'] = f"data:image/png;base64,{image_data}"
                except Exception as e:
                    item['image_data'] = None

            results.append({
                "item": item,
                "score": result['score']
            })

        return JSONResponse({
            "query_image": str(image_path),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up query image
        if image_path.exists():
            image_path.unlink()


@app.post("/api/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    caption: str = Form("")
):
    """ドキュメントまたは画像をアップロード."""
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    # Save file
    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Add to vector store
        if file.filename.endswith(('.png', '.jpg', '.jpeg')):
            # Image
            assistant.vector_store.add_image(
                str(file_path),
                caption=caption or file.filename,
                metadata={"uploaded_at": datetime.now().isoformat()}
            )
        elif file.filename.endswith(('.md', '.txt')):
            # Text document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assistant.vector_store.add_text(
                content,
                metadata={
                    "source": file.filename,
                    "uploaded_at": datetime.now().isoformat()
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save vector store
        assistant.vector_store.save(VECTOR_STORE_PATH)

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "type": "image" if file.filename.endswith(('.png', '.jpg', '.jpeg')) else "text",
            "total_items": len(assistant.vector_store.items)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/build-from-directory")
async def build_from_directory(docs_dir: str = Form(...)):
    """ディレクトリからベクトルストアを構築."""
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    try:
        processor = MultimodalDocumentProcessor(assistant.vector_store)
        processor.process_directory(docs_dir)

        # Save
        assistant.vector_store.save(VECTOR_STORE_PATH)

        return JSONResponse({
            "status": "success",
            "docs_dir": docs_dir,
            "total_items": len(assistant.vector_store.items),
            "message": "Vector store built successfully"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/items")
async def get_items(skip: int = 0, limit: int = 50):
    """ベクトルストア内のアイテム一覧を取得."""
    if assistant is None:
        raise HTTPException(status_code=500, detail="Assistant not initialized")

    items = assistant.vector_store.items[skip:skip + limit]

    # Convert to JSON-friendly format
    result_items = []
    for item in items:
        item_copy = item.copy()

        # Add thumbnail for images
        if item['type'] == 'image' and os.path.exists(item['path']):
            try:
                with open(item['path'], 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                    item_copy['thumbnail'] = f"data:image/png;base64,{image_data}"
            except:
                item_copy['thumbnail'] = None

        result_items.append(item_copy)

    return JSONResponse({
        "items": result_items,
        "total": len(assistant.vector_store.items),
        "skip": skip,
        "limit": limit
    })


def get_html_ui() -> str:
    """HTMLユーザーインターフェースを返す."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Game Dev AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        header p {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .status {
            background: #f8f9fa;
            padding: 15px 30px;
            border-bottom: 1px solid #dee2e6;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
        }

        .tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
        }

        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 1.1em;
            transition: all 0.3s;
        }

        .tab:hover {
            background: #e9ecef;
        }

        .tab.active {
            background: white;
            border-bottom: 3px solid #667eea;
            font-weight: bold;
        }

        .tab-content {
            padding: 30px;
        }

        .tab-panel {
            display: none;
        }

        .tab-panel.active {
            display: block;
        }

        .search-box {
            margin-bottom: 30px;
        }

        .search-input {
            width: 100%;
            padding: 15px 20px;
            font-size: 1.1em;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            transition: border-color 0.3s;
        }

        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .search-options {
            margin-top: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .btn {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .results {
            margin-top: 30px;
        }

        .result-item {
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .result-score {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            margin-bottom: 10px;
        }

        .result-type {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 3px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            margin-left: 10px;
        }

        .result-image {
            max-width: 100%;
            max-height: 400px;
            margin-top: 15px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .result-text {
            margin-top: 10px;
            line-height: 1.6;
        }

        .upload-area {
            border: 3px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9fa;
        }

        .upload-area.dragover {
            border-color: #667eea;
            background: #e7f1ff;
        }

        input[type="file"] {
            display: none;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 1.2em;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 Game Dev AI Assistant</h1>
            <p>Multimodal AI for your game development team</p>
        </header>

        <div class="status">
            <span class="status-badge" id="statusBadge">Loading...</span>
            <span id="statusText"></span>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('search')">🔍 Search</button>
            <button class="tab" onclick="showTab('upload')">📤 Upload</button>
            <button class="tab" onclick="showTab('browse')">📚 Browse</button>
        </div>

        <div class="tab-content">
            <!-- Search Tab -->
            <div id="search" class="tab-panel active">
                <div class="search-box">
                    <input type="text" id="searchInput" class="search-input"
                           placeholder="Search documentation (e.g., 'combat UI health bars')" />

                    <div class="search-options">
                        <label>
                            <input type="radio" name="modality" value="" checked> All
                        </label>
                        <label>
                            <input type="radio" name="modality" value="text"> Text Only
                        </label>
                        <label>
                            <input type="radio" name="modality" value="image"> Images Only
                        </label>
                        <label>
                            Results: <input type="number" id="resultCount" value="5" min="1" max="20" style="width: 60px;">
                        </label>
                        <button class="btn" onclick="search()">Search</button>
                    </div>
                </div>

                <div id="searchResults" class="results"></div>
            </div>

            <!-- Upload Tab -->
            <div id="upload" class="tab-panel">
                <h2>Upload Documents or Images</h2>
                <p style="margin: 20px 0; color: #666;">
                    Upload markdown files (.md), text files (.txt), or images (.png, .jpg) to add to the knowledge base.
                </p>

                <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                    <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                    <p style="font-size: 1.2em; margin-bottom: 10px;">Drop files here or click to browse</p>
                    <p style="color: #666;">Supported: .md, .txt, .png, .jpg, .jpeg</p>
                </div>

                <input type="file" id="fileInput" multiple accept=".md,.txt,.png,.jpg,.jpeg" onchange="uploadFiles(this.files)">

                <div id="uploadResults" style="margin-top: 20px;"></div>
            </div>

            <!-- Browse Tab -->
            <div id="browse" class="tab-panel">
                <h2>Browse Knowledge Base</h2>
                <button class="btn" onclick="loadItems()" style="margin: 20px 0;">Refresh</button>
                <div id="browseResults" class="results"></div>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }

        // Load status
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                document.getElementById('statusBadge').textContent = data.status;
                document.getElementById('statusText').textContent = ` ${data.num_items} items loaded`;
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }

        // Search
        async function search() {
            const query = document.getElementById('searchInput').value;
            const modality = document.querySelector('input[name="modality"]:checked').value;
            const k = parseInt(document.getElementById('resultCount').value);

            if (!query) return;

            const resultsDiv = document.getElementById('searchResults');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query, k, modality: modality || null})
                });

                const data = await response.json();
                displayResults(data.results, resultsDiv);
            } catch (error) {
                resultsDiv.innerHTML = `<div class="result-item">Error: ${error.message}</div>`;
            }
        }

        // Display results
        function displayResults(results, container) {
            if (results.length === 0) {
                container.innerHTML = '<div class="result-item">No results found.</div>';
                return;
            }

            container.innerHTML = results.map((result, index) => {
                const item = result.item;
                const score = (result.score * 100).toFixed(1);

                if (item.type === 'text') {
                    return `
                        <div class="result-item">
                            <span class="result-score">Score: ${score}%</span>
                            <span class="result-type">Text</span>
                            <div class="result-text">${item.content.substring(0, 300)}...</div>
                            <small style="color: #666; display: block; margin-top: 10px;">
                                Source: ${item.metadata?.source || 'Unknown'}
                            </small>
                        </div>
                    `;
                } else {
                    return `
                        <div class="result-item">
                            <span class="result-score">Score: ${score}%</span>
                            <span class="result-type">Image</span>
                            <div><strong>${item.caption}</strong></div>
                            ${item.image_data ? `<img src="${item.image_data}" class="result-image" />` : ''}
                            <small style="color: #666; display: block; margin-top: 10px;">
                                Path: ${item.path}
                            </small>
                        </div>
                    `;
                }
            }).join('');
        }

        // Upload files
        async function uploadFiles(files) {
            const resultsDiv = document.getElementById('uploadResults');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Uploading...</div>';

            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('caption', file.name);

                try {
                    const response = await fetch('/api/upload-document', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    resultsDiv.innerHTML += `<div class="result-item">✓ Uploaded: ${file.name}</div>`;
                } catch (error) {
                    resultsDiv.innerHTML += `<div class="result-item">✗ Error uploading ${file.name}: ${error.message}</div>`;
                }
            }

            loadStatus();
        }

        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            uploadFiles(e.dataTransfer.files);
        });

        // Load items
        async function loadItems() {
            const resultsDiv = document.getElementById('browseResults');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';

            try {
                const response = await fetch('/api/items?limit=20');
                const data = await response.json();

                resultsDiv.innerHTML = data.items.map((item, index) => {
                    if (item.type === 'text') {
                        return `
                            <div class="result-item">
                                <span class="result-type">Text</span>
                                <div class="result-text">${item.content.substring(0, 200)}...</div>
                            </div>
                        `;
                    } else {
                        return `
                            <div class="result-item">
                                <span class="result-type">Image</span>
                                <div><strong>${item.caption}</strong></div>
                                ${item.thumbnail ? `<img src="${item.thumbnail}" class="result-image" />` : ''}
                            </div>
                        `;
                    }
                }).join('');
            } catch (error) {
                resultsDiv.innerHTML = `<div class="result-item">Error: ${error.message}</div>`;
            }
        }

        // Enter key for search
        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') search();
        });

        // Load status on page load
        loadStatus();
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    print("Starting Game Dev AI Assistant Web Server...")
    print("Access at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
