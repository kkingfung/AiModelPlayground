# Deployment Checklist - Game Dev AI Assistant

## 🎯 Goal
Get the standalone web app running for your game development team in **under 1 hour**.

---

## ✅ Pre-Deployment Checklist

### Step 1: Verify Environment (5 minutes)

```bash
# Check Python version (need 3.8+)
python --version

# Check if in correct directory
cd D:/PersonalGameDev/AiModelPlayground/experiments/06_domain_specific_ai
pwd

# List files (should see app.py, requirements_app.txt, etc.)
ls -la
```

**Expected output**: Python 3.8 or higher, all files present

---

### Step 2: Install Dependencies (5-10 minutes)

```bash
# Install all required packages
pip install -r requirements_app.txt

# Verify key packages
python -c "import fastapi; print('✓ FastAPI installed')"
python -c "import torch; print('✓ PyTorch installed')"
python -c "import transformers; print('✓ Transformers installed')"
python -c "import faiss; print('✓ FAISS installed')"
```

**If any errors**:
- Missing package → `pip install <package-name>`
- FAISS GPU issues → Use `pip install faiss-cpu` instead

---

### Step 3: Create Test Data (2 minutes)

```bash
# Create sample multimodal docs
python multimodal_assistant.py --create-sample sample_docs_test

# Verify created
ls -la sample_docs_test/
```

**Expected**: `sample_docs_test/` directory with:
- `combat_system.md`
- `inventory_system.md`
- `movement_system.md`
- `images/` folder with caption files

---

### Step 4: Build Initial Vector Store (2 minutes)

```bash
# Build from sample docs
python multimodal_assistant.py \
    --docs-dir sample_docs_test \
    --build \
    --save vector_store_web
```

**Expected output**:
```
Loading CLIP model...
Processing directory: sample_docs_test
Processed X text chunks and Y images
Saved multimodal vector store to vector_store_web
```

---

### Step 5: Test Vector Store (1 minute)

```bash
# Quick query test
python multimodal_assistant.py \
    --load vector_store_web \
    --query "combat system" \
    --k 3
```

**Expected**: Should return 3 results with scores

---

## 🚀 Launch the Web App

### Step 6: Start Server (30 seconds)

```bash
# Start the web application
python app.py
```

**Expected output**:
```
Starting Game Dev AI Assistant...
Loading existing vector store from vector_store_web
Loaded X items
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Server is now running!** ✅

---

### Step 7: Test Web UI (2 minutes)

**Open browser**: http://localhost:8000

**Check all tabs**:

#### 🔍 Search Tab
1. Type: "combat system"
2. Click "Search"
3. **Expected**: Results with text and images
4. **Verify**: Scores shown, images display

#### 📤 Upload Tab
1. Create a test file: `echo "Test document" > test.txt`
2. Drag `test.txt` to upload area
3. **Expected**: "✓ Uploaded: test.txt"
4. **Verify**: Status shows increased item count

#### 📚 Browse Tab
1. Click "Refresh"
2. **Expected**: List of all items
3. **Verify**: Can see text snippets and image thumbnails

**If all working** → ✅ **Web app is ready!**

---

## 📚 Add Your Real Documentation

### Step 8: Prepare Your Game Docs (5-10 minutes)

Create organized documentation structure:

```bash
# Create game docs directory
mkdir -p game_docs/{design,technical,screenshots,diagrams}

# Example structure:
game_docs/
├── design/
│   ├── combat_system.md
│   ├── progression.md
│   └── economy.md
├── technical/
│   ├── architecture.md
│   └── api_reference.md
├── screenshots/
│   ├── ui_combat.png
│   ├── ui_combat.txt         # Caption
│   ├── ui_inventory.png
│   └── ui_inventory.txt
└── diagrams/
    ├── state_machine.png
    └── state_machine.txt
```

**For ShaderOp specifically**:
```bash
# Option 1: Copy ShaderOp docs
cp -r D:/PersonalGameDev/ShaderOp/Documentation/* game_docs/design/

# Option 2: Symlink (if you want live updates)
ln -s D:/PersonalGameDev/ShaderOp/Documentation game_docs/shaderop_docs
```

---

### Step 9: Add Screenshots with Captions (5 minutes)

For each screenshot, create a caption file:

```bash
# Example: ui_combat.png
# Create: ui_combat.txt with:
echo "Combat UI showing health bars, action buttons (Attack, Defend, Special), turn indicator, and character portraits. Version 1.2, 1920x1080 resolution." > screenshots/ui_combat.txt
```

**Caption template**:
```
[What it shows], [Key UI elements], [Version], [Resolution], [Context]
```

---

### Step 10: Build Production Vector Store (2-5 minutes)

```bash
# Stop the server (Ctrl+C)

# Build from your game docs
python multimodal_assistant.py \
    --docs-dir game_docs \
    --build \
    --save vector_store_web

# Restart server
python app.py
```

**Expected**: Much more items loaded (50-500+ depending on your docs)

---

## 🌐 Make It Accessible to Team

### Option A: Local Network (Easiest, 2 minutes)

```bash
# Find your local IP
ipconfig  # Windows
# or
ifconfig  # Mac/Linux

# Run server on local network
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Share with team**: `http://YOUR_LOCAL_IP:8000`
- Example: `http://192.168.1.100:8000`

**Team access**: Anyone on same WiFi can use it

---

### Option B: Docker (Production-Ready, 10 minutes)

```bash
# Build Docker image
docker build -t gamedev-ai .

# Run container
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/vector_store_web:/app/vector_store_web \
    -v $(pwd)/uploads:/app/uploads \
    --name gamedev-ai-app \
    gamedev-ai

# Check logs
docker logs -f gamedev-ai-app
```

**Access**: http://localhost:8000

**Persistent data**: Vector store and uploads saved in host directories

---

### Option C: Cloud Deployment (Team Remote Access, 30 minutes)

#### Quick Deploy to Free Tier

**Railway.app** (Easiest):
```bash
# 1. Sign up at railway.app
# 2. Install CLI
npm install -g @railway/cli

# 3. Login
railway login

# 4. Deploy
railway init
railway up

# 5. Get URL
railway domain
```

**Render.com** (Also easy):
1. Push code to GitHub
2. Sign up at render.com
3. New Web Service → Connect GitHub repo
4. Select `experiments/06_domain_specific_ai`
5. Build command: `pip install -r requirements_app.txt`
6. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
7. Deploy!

**Access**: `https://your-app.onrender.com`

---

## 📋 Daily Usage Checklist

### For Team Members

**When someone asks "How does X work?"**:
1. Open http://your-server:8000
2. Type question in search box
3. Get relevant docs + screenshots
4. Share results with team

**When adding new documentation**:
1. Go to "Upload" tab
2. Drag new files
3. Done! Immediately searchable

**When looking for UI reference**:
1. Upload reference screenshot
2. Use image search
3. Find similar UI designs

---

## ⚡ Quick Troubleshooting

### Server won't start
```bash
# Check if port 8000 is busy
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# Use different port
uvicorn app:app --port 8001
```

### Can't install dependencies
```bash
# Try with --user flag
pip install --user -r requirements_app.txt

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements_app.txt
```

### Images not showing
- Check file paths are correct
- Ensure images are RGB (not RGBA)
- Verify caption files exist

### Slow search
- First query is slow (loading model)
- Subsequent queries are fast
- Consider using GPU if available

---

## 🎯 Success Criteria

You're ready for team use when:
- ✅ Web UI loads at http://localhost:8000
- ✅ Search returns relevant results
- ✅ Can upload new files
- ✅ Images display correctly
- ✅ Team can access via URL

---

## 📞 Next Steps After Deployment

### Week 1: Gather Feedback
- Ask team to search for common questions
- Note what works, what doesn't
- Identify missing documentation

### Week 2: Improve Content
- Add more screenshots
- Write better captions
- Fill documentation gaps

### Week 3: Enhance Features
- Add authentication (if needed)
- Integrate GPT-4V for answers
- Fine-tune CLIP on your UI

---

## 🆘 Need Help?

**Check**:
1. `DEPLOYMENT_GUIDE.md` - Full deployment manual
2. `MULTIMODAL_GUIDE.md` - How multimodal search works
3. `QUICKSTART.md` - Quick start guide
4. http://localhost:8000/docs - Auto-generated API docs

**Common Issues**:
- Port already in use → Use different port
- CUDA errors → Use CPU (faiss-cpu)
- Module not found → Check requirements installed

---

## ✨ You're Done!

Your team now has:
- 🔍 **Visual search** - Find UI designs by screenshot
- 📚 **Smart documentation** - Search across all docs
- 🎨 **Multimodal AI** - Text + images together
- 🌐 **Web interface** - No installation needed
- 🚀 **Production ready** - Docker + cloud deployment

**Estimated total time**: 30-60 minutes from start to team-ready!

---

**Ready to start? Let's go! 🎮🤖**
