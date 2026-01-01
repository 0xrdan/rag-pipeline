/**
 * RAG Pipeline - Basic Usage Examples
 *
 * Run with: npx ts-node examples/basic-usage.ts
 * Requires: ChromaDB running on localhost:8000
 */

import {
  Pipeline,
  ChunkingService,
  EmbeddingService,
  DocumentInput,
} from '../src';

async function main() {
  console.log('=== RAG Pipeline Examples ===\n');

  // ----------------------------------------------------------------------------
  // Example 1: Basic Pipeline Usage
  // ----------------------------------------------------------------------------
  console.log('--- Example 1: Basic Pipeline ---');

  const pipeline = new Pipeline({
    rag: {
      topK: 3,
      threshold: 0.4,
      enableQueryExpansion: true,
      enableHybridSearch: true,
    },
  });

  try {
    await pipeline.initialize();
    console.log('Pipeline initialized');
  } catch (error) {
    console.error('Failed to initialize. Is ChromaDB running on localhost:8000?');
    console.error('Start with: docker run -p 8000:8000 chromadb/chroma');
    return;
  }

  // Sample documents
  const documents: DocumentInput[] = [
    {
      id: 'doc-1',
      title: 'Introduction to Machine Learning',
      content: `Machine learning (ML) is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly programmed.

        There are three main types of machine learning:
        1. Supervised Learning - learning from labeled data
        2. Unsupervised Learning - finding patterns in unlabeled data
        3. Reinforcement Learning - learning through trial and error

        Common applications include image recognition, natural language processing,
        and recommendation systems.`,
      source: 'tutorials',
      category: 'AI',
      tags: ['ml', 'ai', 'machine-learning'],
    },
    {
      id: 'doc-2',
      title: 'RAG Pipeline Architecture',
      content: `Retrieval-Augmented Generation (RAG) combines the power of large language
        models with external knowledge retrieval.

        The RAG architecture consists of:
        - Document Processing: Chunking documents into semantic pieces
        - Embedding Generation: Converting text to vector representations
        - Vector Storage: Storing embeddings in a vector database like ChromaDB
        - Retrieval: Finding relevant chunks using similarity search
        - Generation: Using retrieved context to generate accurate responses

        RAG helps reduce hallucinations and keeps responses grounded in actual data.`,
      source: 'tutorials',
      category: 'AI',
      tags: ['rag', 'llm', 'architecture'],
    },
    {
      id: 'doc-3',
      title: 'ChromaDB Setup Guide',
      content: `ChromaDB is an open-source vector database designed for AI applications.

        Installation options:
        - Docker: docker run -p 8000:8000 chromadb/chroma
        - Python: pip install chromadb && chroma run
        - Embedded: Use directly in your Python application

        ChromaDB supports multiple distance metrics including cosine similarity,
        L2 distance, and inner product. It's designed for simplicity and integrates
        well with LangChain and other AI frameworks.`,
      source: 'guides',
      category: 'Database',
      tags: ['chromadb', 'vector-database', 'setup'],
    },
  ];

  // Clear and index
  console.log('\nClearing existing data...');
  await pipeline.clearAll();

  console.log('Indexing documents...');
  const chunksIndexed = await pipeline.indexDocuments(documents);
  console.log(`Indexed ${chunksIndexed} chunks from ${documents.length} documents`);

  // Query examples
  console.log('\n--- Query Examples ---\n');

  // Query 1: Direct match
  const q1 = await pipeline.query({
    question: 'What is machine learning?',
  });
  console.log('Q: What is machine learning?');
  console.log(`Found ${q1.chunks.length} chunks, avg similarity: ${q1.stats.avgSimilarity}`);
  console.log(`Top result: "${q1.chunks[0]?.metadata.title}" (${q1.chunks[0]?.score.toFixed(3)})`);

  // Query 2: Synonym expansion (ML -> machine learning)
  console.log('\n');
  const q2 = await pipeline.query({
    question: 'How does ML work?',
  });
  console.log('Q: How does ML work? (tests synonym expansion)');
  console.log(`Found ${q2.chunks.length} chunks, avg similarity: ${q2.stats.avgSimilarity}`);
  console.log(`Top result: "${q2.chunks[0]?.metadata.title}"`);

  // Query 3: With filters
  console.log('\n');
  const q3 = await pipeline.query({
    question: 'How do I set up the database?',
    filters: { source: 'guides' },
  });
  console.log('Q: How do I set up the database? (filtered to guides)');
  console.log(`Found ${q3.chunks.length} chunks`);
  console.log(`Top result: "${q3.chunks[0]?.metadata.title}"`);

  // Print context for LLM
  console.log('\n--- Generated Context (for LLM) ---\n');
  console.log(q1.context.substring(0, 500) + '...');

  // ----------------------------------------------------------------------------
  // Example 2: Chunking Service
  // ----------------------------------------------------------------------------
  console.log('\n--- Example 2: Chunking Service ---');

  const chunker = new ChunkingService({
    maxWordsPerChunk: 100,
    minWordsPerChunk: 20,
  });

  const longDoc: DocumentInput = {
    id: 'long-doc',
    title: 'Long Document',
    content: `This is the first paragraph with some content about topic A.

      This is the second paragraph discussing topic B in more detail.
      It has multiple sentences and covers various aspects.

      This is the third paragraph about topic C.

      And finally, a fourth paragraph wrapping things up.`,
    source: 'test',
  };

  const chunks = chunker.chunkDocument(longDoc);
  console.log(`Document split into ${chunks.length} chunks:`);
  chunks.forEach((chunk, i) => {
    console.log(`  Chunk ${i + 1}: ${chunk.content.substring(0, 50)}...`);
  });

  // ----------------------------------------------------------------------------
  // Example 3: Statistics
  // ----------------------------------------------------------------------------
  console.log('\n--- Example 3: Statistics ---');

  const stats = await pipeline.getStats();
  console.log('Pipeline stats:', stats);

  console.log('\n=== Examples Complete ===');
}

// Run examples
main().catch(console.error);
