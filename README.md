# RAG Pipeline

Production-ready Retrieval-Augmented Generation with ChromaDB, OpenAI embeddings, and hybrid search.

> *This is a standalone extraction from my production portfolio site. See it in action at [danmonteiro.com](https://www.danmonteiro.com).*

---

## The Problem

You're building a RAG system but:

- **Chunking is an afterthought** — splitting on token count loses semantic meaning
- **Pure vector search isn't enough** — missing obvious keyword matches
- **No query understanding** — "ML" doesn't find "machine learning" content
- **Scattered implementation** — embedding, storage, and retrieval are disconnected

## The Solution

RAG Pipeline provides:

- **Semantic chunking** — split documents by meaning, not arbitrary limits
- **Hybrid search** — vector similarity + keyword boosting
- **Query expansion** — automatic synonym handling (ML ↔ machine learning)
- **Unified pipeline** — embed, store, and retrieve in one clean API

```typescript
import { Pipeline } from 'rag-pipeline';

const pipeline = new Pipeline();
await pipeline.initialize();

// Index documents
await pipeline.indexDocuments([
  { id: '1', title: 'ML Guide', content: '...', source: 'docs' }
]);

// Query with hybrid search
const result = await pipeline.query({
  question: "How does machine learning work?"
});

console.log(result.context);  // Formatted context for LLM
console.log(result.chunks);   // Retrieved chunks with scores
```

## Results

From production usage:

| Metric | Vector Only | Hybrid Search |
|--------|-------------|---------------|
| Recall@5 | 72% | 89% |
| Exact match boost | 0% | +20% score |
| Synonym coverage | None | Automatic |

---

## Design Philosophy

### Why Hybrid Search?

Pure vector search has blind spots:

1. **Exact matches matter**: If someone asks about "ChromaDB" and you have a doc titled "ChromaDB Setup Guide", that should rank higher—even if the embedding similarity is similar to other database docs.

2. **Acronyms are tricky**: "ML" and "machine learning" have different embeddings but mean the same thing. Query expansion catches this.

3. **Precision vs recall tradeoff**: Vector search optimizes for semantic similarity. Keyword boost adds precision for obvious matches.

### Chunking Strategy

```
Document → Paragraphs → Sentences (if needed) → Chunks
              ↓
       Respect semantic boundaries
              ↓
       Configurable overlap
```

The chunking service:
- Splits on paragraph boundaries first (semantic units)
- Only breaks paragraphs if they exceed max size
- Preserves metadata for source attribution

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Pipeline                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Chunking   │──│  Embeddings │──│   Vector DB     │ │
│  │  Service    │  │   Service   │  │   (ChromaDB)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         ↓               ↓                  ↓            │
│  ┌─────────────────────────────────────────────────┐   │
│  │               RAG Service                        │   │
│  │  • Query expansion  • Hybrid search  • Context  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
npm install rag-pipeline
```

### 2. Start ChromaDB

```bash
# Using Docker
docker run -p 8000:8000 chromadb/chroma

# Or install locally
pip install chromadb
chroma run
```

### 3. Configure

```bash
export OPENAI_API_KEY="sk-..."
export CHROMA_URL="http://localhost:8000"  # Optional, default
```

### 4. Use

```typescript
import { Pipeline } from 'rag-pipeline';

const pipeline = new Pipeline();
await pipeline.initialize();

// Index your documents
await pipeline.indexDocuments([
  {
    id: 'doc-1',
    title: 'Introduction to RAG',
    content: 'Retrieval-Augmented Generation combines...',
    source: 'tutorials',
    tags: ['rag', 'llm', 'ai'],
  },
]);

// Query
const result = await pipeline.query({
  question: 'What is RAG?',
});

// Use the context with your LLM
const prompt = `Context:\n${result.context}\n\nQuestion: What is RAG?`;
```

---

## API Reference

### Pipeline

The main entry point combining all components.

```typescript
const pipeline = new Pipeline({
  vectorDB: { host: 'http://localhost:8000' },
  embeddings: { model: 'text-embedding-3-large' },
  chunking: { maxWordsPerChunk: 500 },
  rag: { topK: 5, threshold: 0.5 },
});
```

#### Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Connect to ChromaDB |
| `indexDocument(doc)` | Index a single document |
| `indexDocuments(docs)` | Index multiple documents |
| `query(query)` | Retrieve relevant chunks |
| `deleteBySource(source)` | Delete all docs from a source |
| `clearAll()` | Clear all indexed data |
| `getStats()` | Get collection statistics |

### ChunkingService

Semantic document chunking.

```typescript
import { ChunkingService } from 'rag-pipeline';

const chunker = new ChunkingService({
  maxWordsPerChunk: 500,
  minWordsPerChunk: 50,
  overlapWords: 50,
});

const chunks = chunker.chunkDocument({
  id: '1',
  title: 'My Doc',
  content: '...',
  source: 'docs',
});
```

### EmbeddingService

OpenAI embeddings with dimension reduction.

```typescript
import { EmbeddingService } from 'rag-pipeline';

const embeddings = new EmbeddingService({
  model: 'text-embedding-3-large',
  dimensions: 1024,  // Optional reduction
});

const vector = await embeddings.embed('Hello world');
const vectors = await embeddings.embedBatch(['Hello', 'World']);
```

### VectorDatabase

ChromaDB wrapper with clean interface.

```typescript
import { VectorDatabase } from 'rag-pipeline';

const db = new VectorDatabase({
  host: 'http://localhost:8000',
  collectionName: 'my_collection',
  distanceMetric: 'cosine',
});

await db.initialize();
await db.upsertChunks(chunks, embeddings);
const results = await db.similaritySearch(queryEmbedding, 5);
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `CHROMA_URL` | No | `http://localhost:8000` | ChromaDB URL |
| `CHROMA_COLLECTION` | No | `knowledge_base` | Collection name |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model |
| `EMBEDDING_DIMENSIONS` | No | Native | Reduced dimensions |

### RAG Configuration

```typescript
const pipeline = new Pipeline({
  rag: {
    topK: 5,                    // Chunks to retrieve
    threshold: 0.5,             // Minimum similarity
    enableQueryExpansion: true, // Synonym handling
    enableHybridSearch: true,   // Keyword boosting
    synonymMap: {               // Custom synonyms
      'k8s': ['kubernetes', 'Kubernetes'],
    },
  },
});
```

---

## Project Structure

```
rag-pipeline/
├── src/
│   ├── index.ts        # Main exports + Pipeline class
│   ├── vector-db.ts    # ChromaDB integration
│   ├── embeddings.ts   # OpenAI embeddings
│   ├── chunking.ts     # Semantic chunking
│   └── rag.ts          # RAG service (retrieval logic)
├── examples/
│   └── basic-usage.ts
├── docs/
│   └── architecture.md
└── README.md
```

---

## Advanced Usage

### Custom Chunking

```typescript
import { ChunkingService, DocumentInput } from 'rag-pipeline';

const chunker = new ChunkingService();

// Chunk by markdown sections
const chunks = chunker.chunkBySections({
  id: '1',
  title: 'Guide',
  content: '## Section 1\n...\n## Section 2\n...',
  source: 'docs',
});

// Create overview chunk
const overview = chunker.createOverviewChunk(doc, [
  { value: doc.title },
  { label: 'Category', value: doc.category },
  { label: 'Tags', value: doc.tags?.join(', ') },
]);
```

### Direct Component Access

```typescript
const pipeline = new Pipeline();
await pipeline.initialize();

// Get individual services
const embeddings = pipeline.getEmbeddingService();
const vectorDB = pipeline.getVectorDB();
const chunking = pipeline.getChunkingService();
const rag = pipeline.getRAGService();

// Use directly
const vector = await embeddings.embed('test');
```

### Custom Filters

```typescript
// Filter by source
const result = await pipeline.query({
  question: 'How to deploy?',
  filters: { source: 'tutorials' },
});

// Filter by category
const result = await pipeline.query({
  question: 'What is RAG?',
  filters: { category: 'AI' },
});
```

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/add-new-feature`)
3. Make changes with semantic commits
4. Open a PR with clear description

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with [Claude Code](https://claude.ai/code).

```
Co-Authored-By: Claude <noreply@anthropic.com>
```
