"""
RAG Pipeline for Game Development Documentation

This module implements a Retrieval-Augmented Generation system that:
1. Ingests game documentation (markdown, text, code files)
2. Creates embeddings and stores them in a vector database
3. Retrieves relevant documents for user queries
4. Provides context to LLM for answer generation
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Install with: pip install langchain")


class GameDocRAG:
    """RAG system for game documentation."""

    def __init__(
        self,
        docs_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_path: str = "vector_store"
    ):
        """
        Initialize RAG pipeline.

        Args:
            docs_dir: ドキュメントディレクトリのパス
            embedding_model: 使用する埋め込みモデル
            chunk_size: チャンクサイズ（トークン数）
            chunk_overlap: チャンク間のオーバーラップ
            vector_store_path: ベクトルストアの保存パス
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain")

        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path

        # 埋め込みモデルの初期化
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
        )

        self.vectorstore: Optional[FAISS] = None

    def load_documents(self) -> List[Document]:
        """
        ドキュメントディレクトリからすべてのファイルを読み込む.

        Returns:
            ドキュメントのリスト
        """
        print(f"Loading documents from {self.docs_dir}")

        documents = []

        # Markdownファイル
        md_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents.extend(md_loader.load())

        # テキストファイル
        txt_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents.extend(txt_loader.load())

        print(f"Loaded {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントをチャンクに分割.

        Args:
            documents: 分割するドキュメント

        Returns:
            チャンクのリスト
        """
        print(f"Chunking documents (size={self.chunk_size}, overlap={self.chunk_overlap})")

        # セクション境界で分割するセパレータ
        separators = [
            "\n## ",  # Markdown h2
            "\n### ", # Markdown h3
            "\n#### ",# Markdown h4
            "\n\n",   # Paragraphs
            "\n",     # Lines
            " ",      # Words
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        # メタデータの追加
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['char_count'] = len(chunk.page_content)

        return chunks

    def build_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        チャンクからベクトルストアを構築.

        Args:
            chunks: ドキュメントチャンク

        Returns:
            FAISSベクトルストア
        """
        print("Building vector store (this may take a while)...")

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        print(f"Vector store built with {len(chunks)} chunks")
        return vectorstore

    def save_vector_store(self):
        """ベクトルストアをディスクに保存."""
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Build it first.")

        print(f"Saving vector store to {self.vector_store_path}")
        self.vectorstore.save_local(self.vector_store_path)
        print("Vector store saved")

    def load_vector_store(self):
        """保存済みベクトルストアを読み込む."""
        print(f"Loading vector store from {self.vector_store_path}")
        self.vectorstore = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # Note: Only use with trusted data
        )
        print("Vector store loaded")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20
    ) -> List[Document]:
        """
        クエリに関連するドキュメントを検索.

        Args:
            query: 検索クエリ
            k: 返すドキュメント数
            search_type: 検索タイプ ("similarity" or "mmr")
            fetch_k: MMR用のフェッチ数

        Returns:
            関連ドキュメントのリスト
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Build or load one first.")

        print(f"Retrieving documents for query: {query[:50]}...")

        if search_type == "mmr":
            # Maximum Marginal Relevance - 多様性を考慮
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k}
            )
        else:
            # 類似度ベース
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )

        docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(docs)} documents")

        return docs

    def build_from_directory(self):
        """ディレクトリからベクトルストアを構築（フルパイプライン）."""
        # 1. ドキュメント読み込み
        documents = self.load_documents()

        # 2. チャンク分割
        chunks = self.chunk_documents(documents)

        # 3. ベクトルストア構築
        self.vectorstore = self.build_vector_store(chunks)

        # 4. 保存
        self.save_vector_store()

        print("RAG pipeline built successfully!")

    def search_and_display(self, query: str, k: int = 3):
        """
        検索結果を表示.

        Args:
            query: 検索クエリ
            k: 表示するドキュメント数
        """
        docs = self.retrieve(query, k=k)

        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        for i, doc in enumerate(docs, 1):
            print(f"--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"Content:\n{doc.page_content[:300]}...")
            print()


def create_sample_docs(output_dir: str):
    """
    サンプルゲームドキュメントを作成.

    Args:
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # サンプル1: Combat System
    combat_doc = """# Combat System

## Overview

The combat system in our game uses a turn-based approach where players select actions during their turn.

## Actions

### Attack
- **Description**: Deal damage to an enemy
- **Formula**: `damage = attack_power * (1 - defense_modifier) * critical_multiplier`
- **Critical Hit**: 10% chance, deals 2x damage

### Defend
- **Description**: Reduce incoming damage for one turn
- **Effect**: Reduces damage by 50%
- **Duration**: Until next turn

### Special Ability
- **Description**: Use character-specific special abilities
- **Cost**: Consumes mana/energy
- **Cooldown**: 3 turns

## Status Effects

- **Poison**: Deals 5% max HP damage per turn for 3 turns
- **Stun**: Skip next turn
- **Buff**: Increase attack by 25% for 2 turns

## Implementation

```csharp
public class CombatSystem
{
    public float CalculateDamage(Character attacker, Character defender)
    {
        float baseDamage = attacker.AttackPower;
        float defenseModifier = defender.Defense / 100f;
        float critMultiplier = Random.value < 0.1f ? 2f : 1f;

        return baseDamage * (1 - defenseModifier) * critMultiplier;
    }
}
```
"""

    # サンプル2: Inventory System
    inventory_doc = """# Inventory System

## Overview

The inventory system manages player items with categorization and sorting capabilities.

## Item Types

- **Weapon**: Swords, bows, staffs
- **Armor**: Helmets, chest pieces, boots
- **Consumable**: Potions, food, scrolls
- **Quest**: Quest-related items
- **Material**: Crafting materials

## Features

### Sorting

```csharp
// Sort by rarity
InventoryManager.Instance.SortItems(SortType.Rarity);

// Sort by name
InventoryManager.Instance.SortItems(SortType.Name);

// Sort by quantity
InventoryManager.Instance.SortItems(SortType.Quantity);
```

### Stacking

Items of the same type automatically stack up to MaxStackSize (default: 99).

### Weight System

- Each item has a weight value
- Player has maximum carry capacity
- Exceeding capacity reduces movement speed by 25%

## Implementation

The InventoryManager uses a singleton pattern:

```csharp
public class InventoryManager : MonoBehaviour
{
    public static InventoryManager Instance { get; private set; }

    private List<InventorySlot> _slots;

    public void AddItem(Item item)
    {
        // Try to stack with existing item
        var existingSlot = _slots.Find(s => s.Item.ID == item.ID && s.CanStack);

        if (existingSlot != null)
        {
            existingSlot.Quantity++;
        }
        else
        {
            // Add to new slot
            var emptySlot = _slots.Find(s => s.IsEmpty);
            emptySlot.SetItem(item);
        }
    }
}
```
"""

    # サンプル3: Player Movement
    movement_doc = """# Player Movement System

## Movement Speed

Player movement speed is calculated based on several factors:

```
final_speed = base_speed * (1 + agility/100) * terrain_modifier * status_modifier
```

### Base Speed
- Default: 5.0 units/second
- Can be modified by equipment

### Agility Modifier
- Each point of agility increases speed by 1%
- Example: 50 agility = 50% speed increase

### Terrain Modifiers
- **Grass**: 1.0x (normal)
- **Road**: 1.2x (faster)
- **Water**: 0.5x (slower)
- **Mountain**: 0.7x (slower)

### Status Modifiers
- **Haste**: 1.5x speed
- **Slow**: 0.5x speed
- **Encumbered** (over carry capacity): 0.75x speed

## Implementation

```csharp
public class PlayerMovement : MonoBehaviour
{
    [SerializeField] private float _baseSpeed = 5f;

    private float GetCurrentSpeed()
    {
        float agilityMod = 1f + (player.Agility / 100f);
        float terrainMod = GetTerrainModifier();
        float statusMod = GetStatusModifier();

        return _baseSpeed * agilityMod * terrainMod * statusMod;
    }
}
```

## Controls

- **WASD** or **Arrow Keys**: Move character
- **Shift**: Sprint (2x speed, consumes stamina)
- **Space**: Jump
- **Ctrl**: Crouch (0.5x speed, reduced detection)
"""

    # ファイルに書き込み
    with open(os.path.join(output_dir, "combat_system.md"), "w", encoding="utf-8") as f:
        f.write(combat_doc)

    with open(os.path.join(output_dir, "inventory_system.md"), "w", encoding="utf-8") as f:
        f.write(inventory_doc)

    with open(os.path.join(output_dir, "movement_system.md"), "w", encoding="utf-8") as f:
        f.write(movement_doc)

    print(f"Created sample documents in {output_dir}")


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description="RAG Pipeline for Game Documentation")
    parser.add_argument("--docs-dir", type=str, default="sample_docs", help="Documentation directory")
    parser.add_argument("--build", action="store_true", help="Build vector store from documents")
    parser.add_argument("--load", action="store_true", help="Load existing vector store")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--create-samples", action="store_true", help="Create sample documents")
    parser.add_argument("--vector-store", type=str, default="vector_store", help="Vector store path")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve")

    args = parser.parse_args()

    # サンプルドキュメント作成
    if args.create_samples:
        create_sample_docs(args.docs_dir)
        return

    # RAGシステム初期化
    rag = GameDocRAG(
        docs_dir=args.docs_dir,
        vector_store_path=args.vector_store
    )

    # ベクトルストア構築
    if args.build:
        rag.build_from_directory()

    # ベクトルストア読み込み
    if args.load or args.query:
        rag.load_vector_store()

    # クエリ実行
    if args.query:
        rag.search_and_display(args.query, k=args.k)


if __name__ == "__main__":
    main()
