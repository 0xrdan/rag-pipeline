/**
 * RAG Pipeline - Production-ready Retrieval-Augmented Generation
 *
 * A complete RAG system with ChromaDB vector storage, OpenAI embeddings,
 * semantic chunking, and hybrid search capabilities.
 *
 * @packageDocumentation
 */

// Vector Database
export {
  ContentChunk,
  RetrievedChunk,
  VectorDBConfig,
  VectorDatabase,
  getVectorDB,
  createVectorDB,
} from './vector-db';

// Embeddings
export {
  EmbeddingModelConfig,
  EmbeddingServiceConfig,
  EMBEDDING_MODELS,
  EmbeddingService,
  getEmbeddingService,
  createEmbeddingService,
} from './embeddings';

// Chunking
export {
  ChunkingConfig,
  DocumentInput,
  stripMarkdown,
  truncateWords,
  countWords,
  ChunkingService,
  getChunkingService,
  createChunkingService,
} from './chunking';

// RAG Service
export {
  RAGConfig,
  RAGQuery,
  RAGResult,
  RAGResponse,
  RAGService,
  getRAGService,
  createRAGService,
} from './rag';

// =============================================================================
// Convenience: Pipeline Class
// =============================================================================

import { VectorDatabase, ContentChunk, createVectorDB, VectorDBConfig } from './vector-db';
import { EmbeddingService, createEmbeddingService, EmbeddingServiceConfig } from './embeddings';
import { ChunkingService, createChunkingService, ChunkingConfig, DocumentInput } from './chunking';
import { RAGService, createRAGService, RAGConfig, RAGQuery, RAGResult } from './rag';

export interface PipelineConfig {
  vectorDB?: VectorDBConfig;
  embeddings?: EmbeddingServiceConfig;
  chunking?: ChunkingConfig;
  rag?: RAGConfig;
}

/**
 * RAG Pipeline - High-level interface for the complete RAG system
 *
 * @example
 * ```typescript
 * const pipeline = new Pipeline();
 * await pipeline.initialize();
 *
 * // Index documents
 * await pipeline.indexDocuments([
 *   { id: '1', title: 'Doc 1', content: '...', source: 'docs' }
 * ]);
 *
 * // Query
 * const result = await pipeline.query({ question: 'What is...?' });
 * console.log(result.context);
 * ```
 */
export class Pipeline {
  private vectorDB: VectorDatabase;
  private embeddings: EmbeddingService;
  private chunking: ChunkingService;
  private rag: RAGService;
  private initialized: boolean = false;

  constructor(config?: PipelineConfig) {
    this.vectorDB = createVectorDB(config?.vectorDB);
    this.embeddings = createEmbeddingService(config?.embeddings);
    this.chunking = createChunkingService(config?.chunking);
    this.rag = createRAGService(config?.rag);
  }

  /**
   * Initialize the pipeline (connects to vector DB)
   */
  async initialize(): Promise<void> {
    await this.vectorDB.initialize();
    this.initialized = true;
    console.log('[Pipeline] Initialized');
  }

  /**
   * Ensure pipeline is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('Pipeline not initialized. Call initialize() first.');
    }
  }

  /**
   * Index a single document
   */
  async indexDocument(doc: DocumentInput): Promise<number> {
    this.ensureInitialized();

    const chunks = this.chunking.chunkDocument(doc);
    if (chunks.length === 0) {
      console.warn(`[Pipeline] No chunks created for document: ${doc.title}`);
      return 0;
    }

    const embeddings = await this.embeddings.embedBatch(
      chunks.map(c => c.content)
    );

    await this.vectorDB.upsertChunks(chunks, embeddings);
    console.log(`[Pipeline] Indexed ${chunks.length} chunks for: ${doc.title}`);

    return chunks.length;
  }

  /**
   * Index multiple documents
   */
  async indexDocuments(docs: DocumentInput[]): Promise<number> {
    this.ensureInitialized();

    let totalChunks = 0;
    for (const doc of docs) {
      totalChunks += await this.indexDocument(doc);
    }

    console.log(`[Pipeline] Indexed ${totalChunks} total chunks from ${docs.length} documents`);
    return totalChunks;
  }

  /**
   * Query the RAG pipeline
   */
  async query(query: RAGQuery): Promise<RAGResult> {
    this.ensureInitialized();

    // Generate embedding for query
    const expandedQuery = query.question; // RAG service handles expansion
    const queryEmbedding = await this.embeddings.embed(expandedQuery);

    // Search vector DB
    const chunks = await this.vectorDB.similaritySearch(
      queryEmbedding,
      this.rag.getConfig().topK,
      query.filters
    );

    // Filter by threshold
    const threshold = this.rag.getConfig().threshold;
    const filteredChunks = chunks.filter(c => c.score >= threshold);

    // Build context
    const context = filteredChunks.length > 0
      ? filteredChunks.map((c, i) =>
          `[Source ${i + 1}: ${c.metadata.title}]\n${c.content}`
        ).join('\n\n---\n\n')
      : 'No relevant context found.';

    const avgSimilarity = filteredChunks.length > 0
      ? filteredChunks.reduce((sum, c) => sum + c.score, 0) / filteredChunks.length
      : 0;

    return {
      chunks: filteredChunks,
      context,
      stats: {
        chunksRetrieved: filteredChunks.length,
        avgSimilarity: Math.round(avgSimilarity * 100) / 100,
        queryTimeMs: 0, // Would need timing wrapper
      },
    };
  }

  /**
   * Delete documents by source
   */
  async deleteBySource(source: string): Promise<void> {
    this.ensureInitialized();
    await this.vectorDB.deleteBySource(source);
  }

  /**
   * Delete a specific document
   */
  async deleteDocument(source: string, sourceId: string): Promise<void> {
    this.ensureInitialized();
    await this.vectorDB.deleteBySourceId(source, sourceId);
  }

  /**
   * Clear all indexed data
   */
  async clearAll(): Promise<void> {
    this.ensureInitialized();
    await this.vectorDB.clearAll();
  }

  /**
   * Get statistics
   */
  async getStats(): Promise<{
    totalChunks: number;
    bySource: Record<string, number>;
    embeddingModel: string;
    embeddingDimensions: number;
  }> {
    this.ensureInitialized();
    const dbStats = await this.vectorDB.getStats();

    return {
      ...dbStats,
      embeddingModel: this.embeddings.getModel(),
      embeddingDimensions: this.embeddings.getDimensions(),
    };
  }

  /**
   * Get the RAG service for advanced configuration
   */
  getRAGService(): RAGService {
    return this.rag;
  }

  /**
   * Get the embedding service for direct access
   */
  getEmbeddingService(): EmbeddingService {
    return this.embeddings;
  }

  /**
   * Get the chunking service for direct access
   */
  getChunkingService(): ChunkingService {
    return this.chunking;
  }

  /**
   * Get the vector database for direct access
   */
  getVectorDB(): VectorDatabase {
    return this.vectorDB;
  }
}

/**
 * Create a pipeline instance
 */
export function createPipeline(config?: PipelineConfig): Pipeline {
  return new Pipeline(config);
}

export default Pipeline;
