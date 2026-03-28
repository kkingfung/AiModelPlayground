"""
Complete Game Development AI Assistant

Combines RAG (retrieval) with fine-tuned LLM (generation) to create a domain-specific
AI assistant for game development teams.
"""

import os
import argparse
from typing import List, Dict, Any, Optional
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class GameDevAssistant:
    """
    ゲーム開発AI アシスタント.

    RAGとファインチューニングを組み合わせたハイブリッドシステム:
    1. RAG: ドキュメント検索で関連情報を取得
    2. Fine-tuned LLM: ゲーム開発ドメインに適応したモデルで回答生成
    """

    def __init__(
        self,
        vector_store_path: str = "vector_store",
        lora_path: Optional[str] = None,
        base_model: str = "meta-llama/Llama-2-7b-chat-hf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True
    ):
        """
        初期化.

        Args:
            vector_store_path: ベクトルストアのパス
            lora_path: LoRAアダプターのパス（Noneの場合はベースモデルのみ）
            base_model: ベースモデル名
            embedding_model: 埋め込みモデル
            use_gpu: GPU使用するか
        """
        self.vector_store_path = vector_store_path
        self.lora_path = lora_path
        self.base_model = base_model
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"Initializing Game Dev Assistant (device: {self.device})")

        # RAGコンポーネント
        if LANGCHAIN_AVAILABLE:
            self._init_rag(embedding_model)
        else:
            print("Warning: LangChain not available. RAG disabled.")
            self.vectorstore = None

        # LLMコンポーネント
        if TRANSFORMERS_AVAILABLE:
            self._init_llm()
        else:
            print("Warning: Transformers not available. LLM disabled.")
            self.model = None
            self.tokenizer = None

    def _init_rag(self, embedding_model: str):
        """RAGコンポーネント初期化."""
        print(f"Loading vector store from {self.vector_store_path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )

        try:
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            self.vectorstore = None

    def _init_llm(self):
        """LLMコンポーネント初期化."""
        print(f"Loading model: {self.base_model}")

        # トークナイザー
        if self.lora_path:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        # モデル
        if self.lora_path:
            # LoRAアダプター使用
            print(f"Loading with LoRA adapters from {self.lora_path}")
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, self.lora_path)
        else:
            # ベースモデルのみ
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        print("Model loaded successfully")

    def retrieve_context(self, query: str, k: int = 3) -> List[Document]:
        """
        クエリに関連するドキュメントを検索.

        Args:
            query: 検索クエリ
            k: 返すドキュメント数

        Returns:
            関連ドキュメントのリスト
        """
        if self.vectorstore is None:
            return []

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4}
        )

        docs = retriever.get_relevant_documents(query)
        return docs

    def format_prompt(self, question: str, context_docs: List[Document]) -> str:
        """
        プロンプトをフォーマット.

        Args:
            question: ユーザーの質問
            context_docs: コンテキストドキュメント

        Returns:
            フォーマットされたプロンプト
        """
        # コンテキストを結合
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""You are a game development expert assistant. Use the following documentation to answer the question. If the documentation doesn't contain the answer, say so.

Documentation:
{context}

Question: {question}

Answer based on the documentation above:"""

        return prompt

    def generate_answer(
        self,
        prompt: str,
        max_length: int = 300,
        temperature: float = 0.7
    ) -> str:
        """
        回答を生成.

        Args:
            prompt: 入力プロンプト
            max_length: 最大長
            temperature: 温度パラメータ

        Returns:
            生成された回答
        """
        if self.model is None:
            return "Error: Model not loaded"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # プロンプト部分を除去
        answer = generated[len(prompt):].strip()

        return answer

    def ask(
        self,
        question: str,
        use_rag: bool = True,
        k: int = 3,
        max_length: int = 300,
        temperature: float = 0.7,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        質問に回答.

        Args:
            question: ユーザーの質問
            use_rag: RAGを使用するか
            k: 検索するドキュメント数
            max_length: 最大生成長
            temperature: 温度パラメータ
            verbose: 詳細情報を表示するか

        Returns:
            回答と関連情報の辞書
        """
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "context_used": use_rag
        }

        # 1. RAGで関連ドキュメント検索
        context_docs = []
        if use_rag and self.vectorstore:
            if verbose:
                print(f"Retrieving relevant documents...")
            context_docs = self.retrieve_context(question, k=k)
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in context_docs
            ]

        # 2. プロンプト作成
        if context_docs:
            prompt = self.format_prompt(question, context_docs)
        else:
            # RAG無しの場合
            prompt = f"""You are a game development expert. Answer the following question:

Question: {question}

Answer:"""

        if verbose:
            print(f"\nPrompt:\n{prompt}\n")
            print("Generating answer...")

        # 3. LLMで回答生成
        answer = self.generate_answer(
            prompt,
            max_length=max_length,
            temperature=temperature
        )

        result["answer"] = answer

        return result

    def display_result(self, result: Dict[str, Any]):
        """
        結果を表示.

        Args:
            result: ask()の戻り値
        """
        print(f"\n{'='*80}")
        print(f"Question: {result['question']}")
        print(f"{'='*80}\n")

        print(f"Answer:\n{result['answer']}\n")

        if result['sources']:
            print(f"{'─'*80}")
            print("Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n[{i}] {source['metadata'].get('source', 'Unknown')}")
                print(f"{source['content']}")

        print(f"{'='*80}\n")


def create_simple_rag_only_assistant(vector_store_path: str) -> 'SimpleRAGAssistant':
    """
    RAGのみのシンプルなアシスタントを作成（LLM不要）.

    Args:
        vector_store_path: ベクトルストアのパス

    Returns:
        SimpleRAGAssistant
    """
    return SimpleRAGAssistant(vector_store_path)


class SimpleRAGAssistant:
    """
    RAGのみのシンプルなアシスタント.

    LLMを使わず、検索結果のみを返す。
    LLM APIと組み合わせて使う場合や、LLMが重い場合に有用。
    """

    def __init__(self, vector_store_path: str):
        """初期化."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        検索実行.

        Args:
            query: 検索クエリ
            k: 結果数

        Returns:
            検索結果のリスト
        """
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )

        docs = retriever.get_relevant_documents(query)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0)
            }
            for doc in docs
        ]


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description="Game Development AI Assistant")
    parser.add_argument("--vector-store", type=str, default="vector_store", help="Vector store path")
    parser.add_argument("--lora", type=str, help="LoRA adapter path (optional)")
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rag-only", action="store_true", help="Use RAG only (no LLM)")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # RAGのみモード
    if args.rag_only:
        print("Using RAG-only mode (no LLM generation)")
        assistant = create_simple_rag_only_assistant(args.vector_store)

        if args.question:
            results = assistant.search(args.question, k=args.k)
            print(f"\nSearch results for: {args.question}\n")
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result['source']}")
                print(f"{result['content']}\n")
        return

    # フルモード（RAG + LLM）
    assistant = GameDevAssistant(
        vector_store_path=args.vector_store,
        lora_path=args.lora
    )

    # インタラクティブモード
    if args.interactive:
        print("\nGame Dev AI Assistant - Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            question = input("Question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            result = assistant.ask(question, verbose=args.verbose, k=args.k)
            assistant.display_result(result)

    # 単一質問モード
    elif args.question:
        result = assistant.ask(args.question, verbose=args.verbose, k=args.k)
        assistant.display_result(result)

    else:
        print("Error: Use --question or --interactive mode")


if __name__ == "__main__":
    main()
