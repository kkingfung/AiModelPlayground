"""
Multimodal Game Development AI Assistant

Combines text and image understanding for comprehensive game dev support:
- UI/UX screenshot analysis
- Design diagram understanding
- Bug report visual analysis
- Asset documentation with images
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import base64

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install transformers pillow")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")


class MultimodalVectorStore:
    """
    テキストと画像を統合したベクトルストア.

    CLIPを使用して、テキストと画像を同じ埋め込み空間にマッピング。
    """

    def __init__(self, dimension: int = 512, model_name: str = "openai/clip-vit-base-patch32"):
        """
        初期化.

        Args:
            dimension: 埋め込みの次元数
            model_name: CLIPモデル名
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP required. Install: pip install transformers pillow")

        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install: pip install faiss-cpu")

        self.dimension = dimension
        self.model_name = model_name

        # Load CLIP
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Vector index (Inner Product = cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

        # Metadata storage
        self.items: List[Dict[str, Any]] = []

    def embed_text(self, text: str) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換.

        Args:
            text: 入力テキスト

        Returns:
            正規化された埋め込みベクトル
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Handle both old and new transformers API
            if hasattr(outputs, 'last_hidden_state'):
                # New API returns BaseModelOutputWithPooling
                text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
            else:
                # Old API returns tensor directly
                text_features = outputs

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0]

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        画像を埋め込みベクトルに変換.

        Args:
            image_path: 画像ファイルパス

        Returns:
            正規化された埋め込みベクトル
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Handle both old and new transformers API
            if hasattr(outputs, 'last_hidden_state'):
                # New API returns BaseModelOutputWithPooling
                image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
            else:
                # Old API returns tensor directly
                image_features = outputs

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def add_text(self, text: str, metadata: Optional[Dict] = None):
        """
        テキストを追加.

        Args:
            text: テキスト内容
            metadata: メタデータ
        """
        embedding = self.embed_text(text)
        self.index.add(np.array([embedding]))

        self.items.append({
            "type": "text",
            "content": text,
            "metadata": metadata or {}
        })

    def add_image(self, image_path: str, caption: str = "", metadata: Optional[Dict] = None):
        """
        画像を追加.

        Args:
            image_path: 画像ファイルパス
            caption: キャプション
            metadata: メタデータ
        """
        embedding = self.embed_image(image_path)
        self.index.add(np.array([embedding]))

        self.items.append({
            "type": "image",
            "path": image_path,
            "caption": caption,
            "metadata": metadata or {}
        })

    def search(
        self,
        query: str,
        k: int = 5,
        modality_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        テキストクエリで検索（テキストと画像の両方から）.

        Args:
            query: 検索クエリ
            k: 返す結果数
            modality_filter: "text", "image", または None（両方）

        Returns:
            検索結果のリスト
        """
        query_embedding = self.embed_text(query)

        # Search more than k to allow for filtering
        search_k = k * 3 if modality_filter else k

        distances, indices = self.index.search(np.array([query_embedding]), search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue

            item = self.items[idx]

            # Filter by modality
            if modality_filter and item["type"] != modality_filter:
                continue

            results.append({
                "item": item,
                "score": float(dist)
            })

            if len(results) >= k:
                break

        return results

    def search_by_image(
        self,
        image_path: str,
        k: int = 5,
        modality_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        画像クエリで検索.

        Args:
            image_path: クエリ画像パス
            k: 返す結果数
            modality_filter: "text", "image", または None

        Returns:
            検索結果のリスト
        """
        query_embedding = self.embed_image(image_path)

        search_k = k * 3 if modality_filter else k
        distances, indices = self.index.search(np.array([query_embedding]), search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue

            item = self.items[idx]

            if modality_filter and item["type"] != modality_filter:
                continue

            results.append({
                "item": item,
                "score": float(dist)
            })

            if len(results) >= k:
                break

        return results

    def save(self, directory: str):
        """ベクトルストアを保存."""
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        # Save items metadata
        with open(os.path.join(directory, "items.json"), "w", encoding="utf-8") as f:
            json.dump(self.items, f, indent=2, ensure_ascii=False)

        # Save config
        config = {
            "dimension": self.dimension,
            "model_name": self.model_name,
            "num_items": len(self.items)
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved multimodal vector store to {directory}")

    def load(self, directory: str):
        """ベクトルストアを読み込み."""
        # Load config
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        # Load items
        with open(os.path.join(directory, "items.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)

        print(f"Loaded multimodal vector store from {directory}")
        print(f"  Items: {len(self.items)}")


class MultimodalDocumentProcessor:
    """マルチモーダルドキュメント処理（テキスト + 画像）."""

    def __init__(self, vector_store: MultimodalVectorStore):
        """
        初期化.

        Args:
            vector_store: マルチモーダルベクトルストア
        """
        self.vector_store = vector_store

    def process_directory(self, docs_dir: str):
        """
        ディレクトリ内のドキュメントと画像を処理.

        Args:
            docs_dir: ドキュメントディレクトリ
        """
        docs_path = Path(docs_dir)

        print(f"Processing directory: {docs_dir}")

        # Process text files
        text_count = 0
        for ext in ["*.md", "*.txt"]:
            for file_path in docs_path.rglob(ext):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Simple chunking (split by paragraphs)
                chunks = self._chunk_text(content)

                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:
                        continue

                    self.vector_store.add_text(
                        chunk,
                        metadata={
                            "source": str(file_path),
                            "chunk_id": i,
                            "type": "document"
                        }
                    )
                    text_count += 1

        # Process images
        image_count = 0
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for image_path in docs_path.rglob(ext):
                # Try to find caption file
                caption = self._get_caption(image_path)

                self.vector_store.add_image(
                    str(image_path),
                    caption=caption,
                    metadata={
                        "source": str(image_path),
                        "type": "image"
                    }
                )
                image_count += 1

        print(f"Processed {text_count} text chunks and {image_count} images")

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """テキストをチャンクに分割."""
        # Split by double newline (paragraphs)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para.split())

            if current_size + para_size > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _get_caption(self, image_path: Path) -> str:
        """画像のキャプションを取得（.txtファイルまたはファイル名）."""
        caption_file = image_path.with_suffix(".txt")

        if caption_file.exists():
            with open(caption_file, "r", encoding="utf-8") as f:
                return f.read().strip()

        # Fallback: filename as caption
        return image_path.stem.replace("_", " ").replace("-", " ").title()


class MultimodalGameDevAssistant:
    """マルチモーダルゲーム開発アシスタント."""

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        初期化.

        Args:
            vector_store_path: ベクトルストアのパス（Noneの場合は新規作成）
        """
        self.vector_store = MultimodalVectorStore()

        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store.load(vector_store_path)

    def ask(
        self,
        query: str,
        query_image: Optional[str] = None,
        k: int = 5,
        modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        質問に回答（テキストまたは画像クエリ）.

        Args:
            query: テキストクエリ
            query_image: 画像クエリパス（オプション）
            k: 検索結果数
            modality: フィルター（"text", "image", None）

        Returns:
            検索結果
        """
        if query_image:
            # Image query
            results = self.vector_store.search_by_image(query_image, k=k, modality_filter=modality)
        else:
            # Text query
            results = self.vector_store.search(query, k=k, modality_filter=modality)

        return {
            "query": query,
            "query_image": query_image,
            "results": results
        }

    def display_results(self, response: Dict[str, Any]):
        """結果を表示."""
        print(f"\n{'='*80}")
        print(f"Query: {response['query']}")
        if response.get('query_image'):
            print(f"Query Image: {response['query_image']}")
        print(f"{'='*80}\n")

        for i, result in enumerate(response['results'], 1):
            item = result['item']
            score = result['score']

            print(f"[{i}] Score: {score:.3f}")

            if item['type'] == 'text':
                print(f"Type: Text Document")
                print(f"Source: {item['metadata'].get('source', 'Unknown')}")
                print(f"Content: {item['content'][:200]}...")
            else:
                print(f"Type: Image")
                print(f"Path: {item['path']}")
                print(f"Caption: {item['caption']}")

            print()


def create_sample_multimodal_docs(output_dir: str):
    """サンプルマルチモーダルドキュメントを作成."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Sample text doc
    combat_doc = """# Combat System

## UI Layout

The combat UI consists of:
- Health bars at the top
- Action buttons at the bottom (Attack, Defend, Special)
- Turn indicator on the right
- Character portraits on the left

See combat_ui.png for reference.

## Combat Flow

1. Player selects action
2. Animation plays
3. Damage calculation
4. Enemy turn
5. Repeat until victory/defeat
"""

    with open(os.path.join(output_dir, "combat_system.md"), "w", encoding="utf-8") as f:
        f.write(combat_doc)

    # Create placeholder image metadata (you'd replace with real screenshots)
    image_captions = {
        "combat_ui.png": "Combat UI showing health bars, action buttons (Attack, Defend, Special), and turn indicator",
        "inventory_grid.png": "Inventory screen with 6x4 grid layout, item icons and quantities",
        "state_machine_diagram.png": "Combat state machine diagram: Idle -> PlayerTurn -> EnemyTurn -> Victory/Defeat"
    }

    for filename, caption in image_captions.items():
        caption_path = os.path.join(output_dir, "images", filename.replace(".png", ".txt"))
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

    print(f"Created sample multimodal docs in {output_dir}")
    print("Note: Add actual .png image files to images/ directory")


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description="Multimodal Game Dev Assistant")
    parser.add_argument("--docs-dir", type=str, help="Documentation directory with text and images")
    parser.add_argument("--build", action="store_true", help="Build vector store from docs")
    parser.add_argument("--load", type=str, help="Load existing vector store")
    parser.add_argument("--save", type=str, help="Save vector store path")
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--query-image", type=str, help="Image query path")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--modality", type=str, choices=["text", "image"], help="Filter by modality")
    parser.add_argument("--create-sample", type=str, help="Create sample docs directory")

    args = parser.parse_args()

    # Create sample docs
    if args.create_sample:
        create_sample_multimodal_docs(args.create_sample)
        return

    # Initialize assistant
    assistant = MultimodalGameDevAssistant(vector_store_path=args.load)

    # Build vector store
    if args.build:
        if not args.docs_dir:
            print("Error: --docs-dir required for --build")
            return

        processor = MultimodalDocumentProcessor(assistant.vector_store)
        processor.process_directory(args.docs_dir)

        # Save if path provided
        if args.save:
            assistant.vector_store.save(args.save)

    # Query
    if args.query or args.query_image:
        response = assistant.ask(
            query=args.query or "",
            query_image=args.query_image,
            k=args.k,
            modality=args.modality
        )

        assistant.display_results(response)


if __name__ == "__main__":
    main()
