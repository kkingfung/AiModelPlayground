# RAG Patterns - Quick Reference

Retrieval-Augmented Generation (RAG) patterns for building domain-specific AI systems.

## Core RAG Pipeline

```python
# 1. Document Loading
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("docs/", glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

# 2. Text Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(documents)

# 3. Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector Store
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vector_store")

# 5. Retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 20}
)
relevant_docs = retriever.get_relevant_documents("query")

# 6. Generation (with LLM)
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "How does X work?"})
```

## Embedding Models

### Fast (CPU-friendly)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # 384 dimensions, ~80MB
)
```

### Accurate
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    # 768 dimensions, ~420MB
)
```

### Multilingual
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## Chunking Strategies

### By Tokens (for LLM context)
```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
```

### By Semantic Sections
```python
# Markdown-aware
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n"]
)
```

### By Code Blocks
```python
# For code documentation
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)
```

## Vector Databases

### FAISS (Local, Fast)
```python
from langchain.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(docs, embeddings)

# Save
vectorstore.save_local("vector_store")

# Load
vectorstore = FAISS.load_local("vector_store", embeddings)
```

### Chroma (Persistent, Feature-rich)
```python
from langchain.vectorstores import Chroma

# Create with persistence
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Load
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### Pinecone (Cloud, Scalable)
```python
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

vectorstore = Pinecone.from_documents(
    docs,
    embeddings,
    index_name="my-index"
)
```

## Retrieval Strategies

### Similarity Search
```python
# Basic similarity
docs = vectorstore.similarity_search("query", k=5)

# With scores
docs_with_scores = vectorstore.similarity_search_with_score("query", k=5)
```

### MMR (Maximum Marginal Relevance)
```python
# Balances relevance with diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,        # Final results
        "fetch_k": 20  # Initial candidates
    }
)
```

### Metadata Filtering
```python
# Filter by metadata
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "game_mechanics.md"}
    }
)
```

### Hybrid Search (Semantic + Keyword)
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25 (keyword-based)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Semantic
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine (50/50 weight)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

## Prompt Templates

### Basic RAG Prompt
```python
template = """Use the following context to answer the question.
If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
```

### With Citations
```python
template = """Use the following context to answer the question.
Cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer with citations:"""
```

### Strict Context-Only
```python
template = """Answer ONLY based on the context below.
If the context doesn't contain the answer, say "I don't have enough information."

Context: {context}

Question: {question}

Answer based ONLY on the context:"""
```

## Evaluation Metrics

### Retrieval Quality
```python
def evaluate_retrieval(queries, expected_doc_ids, retriever):
    """Evaluate retrieval accuracy."""
    precision_scores = []
    recall_scores = []

    for query, expected in zip(queries, expected_doc_ids):
        retrieved = retriever.get_relevant_documents(query)
        retrieved_ids = {doc.metadata['id'] for doc in retrieved}
        expected_set = set(expected)

        # Precision: relevant retrieved / total retrieved
        if len(retrieved_ids) > 0:
            precision = len(retrieved_ids & expected_set) / len(retrieved_ids)
        else:
            precision = 0

        # Recall: relevant retrieved / total relevant
        if len(expected_set) > 0:
            recall = len(retrieved_ids & expected_set) / len(expected_set)
        else:
            recall = 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    return {
        'precision': sum(precision_scores) / len(precision_scores),
        'recall': sum(recall_scores) / len(recall_scores)
    }
```

### Response Quality
```python
from rouge_score import rouge_scorer

def evaluate_generation(generated, references):
    """Evaluate generated text quality."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    scores = []
    for gen, ref in zip(generated, references):
        score = scorer.score(ref, gen)
        scores.append(score)

    return scores
```

## Optimization Tips

### 1. Chunk Size
- **Small (200-500)**: Better precision, more results
- **Medium (500-1000)**: Balanced
- **Large (1000-2000)**: More context, fewer results

### 2. Overlap
- Use 10-20% of chunk size
- Ensures context continuity

### 3. Embedding Caching
```python
# Cache embeddings to avoid recomputation
import joblib

# Save
joblib.dump(embeddings, "embeddings.pkl")

# Load
embeddings = joblib.load("embeddings.pkl")
```

### 4. Reranking
```python
from sentence_transformers import CrossEncoder

# Use cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, docs, top_k=3):
    """Rerank retrieved documents."""
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    # Sort by score
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]
```

## Common Issues

### Issue: Poor Retrieval
**Solution**:
- Try different embedding models
- Adjust chunk size/overlap
- Add metadata filtering
- Use hybrid search

### Issue: Hallucinations
**Solution**:
- Use stricter prompts
- Increase retrieval quality
- Add "I don't know" option
- Return source citations

### Issue: Slow Performance
**Solution**:
- Cache embeddings
- Use FAISS GPU index
- Implement query caching
- Batch processing

### Issue: Out of Context
**Solution**:
- Increase chunk size
- Increase overlap
- Retrieve more documents (higher k)
- Use reranking

## Complete Example

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 1. Load docs
loader = DirectoryLoader("game_docs/", glob="**/*.md")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. QA chain
qa = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 6. Ask
result = qa({"query": "How does combat work?"})
print(result["result"])
```

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/modules/data_connection/
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Chroma**: https://www.trychroma.com/
