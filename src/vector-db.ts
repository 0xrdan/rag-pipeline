/**
 * Vector Database - ChromaDB integration for similarity search
 *
 * Provides a clean interface for storing and retrieving document chunks
 * with vector embeddings using ChromaDB.
 *
 * @module vector-db
 */

import { ChromaClient, Collection } from 'chromadb';

// =============================================================================
// Types
// =============================================================================

export interface ContentChunk {
  id: string;
  content: string;
  metadata: {
    source: string;           // Content type (e.g., 'article', 'document')
    sourceId: string;         // ID within the source
    title: string;
    category?: string;
    tags?: string[];          // Converted to string for ChromaDB storage
    url?: string;
    chunkIndex: number;
    totalChunks: number;
    [key: string]: any;       // Allow custom metadata
  };
}

export interface RetrievedChunk extends ContentChunk {
  score: number;              // Similarity score (0-1)
}

export interface VectorDBConfig {
  /** ChromaDB host URL (default: http://localhost:8000) */
  host?: string;
  /** Collection name (default: knowledge_base) */
  collectionName?: string;
  /** Distance metric (default: cosine) */
  distanceMetric?: 'cosine' | 'l2' | 'ip';
}

// =============================================================================
// Vector Database Class
// =============================================================================

export class VectorDatabase {
  private client: ChromaClient;
  private collection: Collection | null = null;
  private collectionName: string;
  private distanceMetric: string;

  constructor(config?: VectorDBConfig) {
    const chromaUrl = config?.host || process.env.CHROMA_URL || 'http://localhost:8000';
    this.collectionName = config?.collectionName || process.env.CHROMA_COLLECTION || 'knowledge_base';
    this.distanceMetric = config?.distanceMetric || 'cosine';

    try {
      const url = new URL(chromaUrl);
      this.client = new ChromaClient({
        host: url.hostname,
        port: parseInt(url.port) || 8000,
        ssl: url.protocol === 'https:',
      });
    } catch {
      // Fallback for simple host:port format
      this.client = new ChromaClient({
        host: 'localhost',
        port: 8000,
      });
    }
  }

  /**
   * Initialize or get the ChromaDB collection
   */
  async initialize(): Promise<void> {
    try {
      this.collection = await this.client.getCollection({
        name: this.collectionName,
      });

      const count = await this.collection.count();
      console.log(`[VectorDB] Connected to collection: ${this.collectionName} (${count} chunks)`);
    } catch (error: any) {
      console.log(`[VectorDB] Creating new collection: ${this.collectionName}`);

      this.collection = await this.client.createCollection({
        name: this.collectionName,
        metadata: {
          'hnsw:space': this.distanceMetric,
        },
      });
      console.log(`[VectorDB] Created collection: ${this.collectionName}`);
    }
  }

  /**
   * Ensure collection is initialized
   */
  private ensureCollection(): Collection {
    if (!this.collection) {
      throw new Error('VectorDB not initialized. Call initialize() first.');
    }
    return this.collection;
  }

  /**
   * Add or update content chunks with embeddings
   */
  async upsertChunks(chunks: ContentChunk[], embeddings: number[][]): Promise<void> {
    const collection = this.ensureCollection();

    const ids = chunks.map(chunk => chunk.id);
    const documents = chunks.map(chunk => chunk.content);

    // Convert metadata to ChromaDB-compatible format (no arrays)
    const metadatas = chunks.map(chunk => {
      const meta: Record<string, any> = {
        source: chunk.metadata.source,
        sourceId: chunk.metadata.sourceId,
        title: chunk.metadata.title,
        chunkIndex: chunk.metadata.chunkIndex,
        totalChunks: chunk.metadata.totalChunks,
      };

      // Convert arrays to comma-separated strings
      if (chunk.metadata.tags) {
        meta.tags = chunk.metadata.tags.join(', ');
      }
      if (chunk.metadata.category) {
        meta.category = chunk.metadata.category;
      }
      if (chunk.metadata.url) {
        meta.url = chunk.metadata.url;
      }

      // Add any custom metadata
      for (const [key, value] of Object.entries(chunk.metadata)) {
        if (!['source', 'sourceId', 'title', 'tags', 'category', 'url', 'chunkIndex', 'totalChunks'].includes(key)) {
          if (Array.isArray(value)) {
            meta[key] = value.join(', ');
          } else if (typeof value !== 'object') {
            meta[key] = value;
          }
        }
      }

      return meta;
    });

    await collection.upsert({
      ids,
      embeddings,
      documents,
      metadatas: metadatas as any,
    });

    console.log(`[VectorDB] Upserted ${chunks.length} chunks`);
  }

  /**
   * Perform similarity search
   */
  async similaritySearch(
    queryEmbedding: number[],
    topK: number = 5,
    filters?: Record<string, any>
  ): Promise<RetrievedChunk[]> {
    const collection = this.ensureCollection();

    const results = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: topK,
      where: filters,
    });

    const chunks: RetrievedChunk[] = [];

    if (results.ids[0] && results.documents[0] && results.metadatas[0] && results.distances[0]) {
      for (let i = 0; i < results.ids[0].length; i++) {
        // Convert distance to similarity score
        // With cosine distance: similarity = 1 - distance
        const distance = results.distances[0][i] as number;
        const similarity = Math.max(0, 1 - distance);

        const chromaMeta = results.metadatas[0][i] as any;

        chunks.push({
          id: results.ids[0][i] as string,
          content: results.documents[0][i] as string,
          metadata: {
            source: chromaMeta.source,
            sourceId: chromaMeta.sourceId,
            title: chromaMeta.title,
            category: chromaMeta.category || undefined,
            tags: chromaMeta.tags
              ? chromaMeta.tags.split(', ').filter((t: string) => t.length > 0)
              : undefined,
            url: chromaMeta.url || undefined,
            chunkIndex: chromaMeta.chunkIndex,
            totalChunks: chromaMeta.totalChunks,
          },
          score: similarity,
        });
      }
    }

    return chunks;
  }

  /**
   * Delete chunks by source ID
   */
  async deleteBySourceId(source: string, sourceId: string): Promise<void> {
    const collection = this.ensureCollection();

    const results = await collection.get({
      where: {
        $and: [
          { source: { $eq: source } },
          { sourceId: { $eq: sourceId } },
        ],
      },
    });

    if (results.ids && results.ids.length > 0) {
      await collection.delete({ ids: results.ids });
      console.log(`[VectorDB] Deleted ${results.ids.length} chunks for ${source}:${sourceId}`);
    }
  }

  /**
   * Delete chunks by source type
   */
  async deleteBySource(source: string): Promise<void> {
    const collection = this.ensureCollection();

    const results = await collection.get({
      where: { source },
    });

    if (results.ids && results.ids.length > 0) {
      await collection.delete({ ids: results.ids });
      console.log(`[VectorDB] Deleted ${results.ids.length} chunks for source: ${source}`);
    }
  }

  /**
   * Get collection statistics
   */
  async getStats(): Promise<{
    totalChunks: number;
    bySource: Record<string, number>;
  }> {
    const collection = this.ensureCollection();
    const count = await collection.count();

    // Get all unique sources
    const allDocs = await collection.get({});
    const bySource: Record<string, number> = {};

    if (allDocs.metadatas) {
      for (const meta of allDocs.metadatas) {
        const source = (meta as any)?.source || 'unknown';
        bySource[source] = (bySource[source] || 0) + 1;
      }
    }

    return { totalChunks: count, bySource };
  }

  /**
   * Clear all data and recreate collection
   */
  async clearAll(): Promise<void> {
    try {
      await this.client.deleteCollection({ name: this.collectionName });
      console.log(`[VectorDB] Deleted collection: ${this.collectionName}`);
    } catch {
      console.log(`[VectorDB] Collection ${this.collectionName} did not exist`);
    }

    this.collection = await this.client.createCollection({
      name: this.collectionName,
      metadata: {
        'hnsw:space': this.distanceMetric,
      },
    });
    console.log(`[VectorDB] Created fresh collection: ${this.collectionName}`);
  }

  /**
   * Get the collection name
   */
  getCollectionName(): string {
    return this.collectionName;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

let vectorDBInstance: VectorDatabase | null = null;

export async function getVectorDB(config?: VectorDBConfig): Promise<VectorDatabase> {
  if (!vectorDBInstance) {
    vectorDBInstance = new VectorDatabase(config);
    await vectorDBInstance.initialize();
  }
  return vectorDBInstance;
}

export function createVectorDB(config?: VectorDBConfig): VectorDatabase {
  return new VectorDatabase(config);
}

export default VectorDatabase;
