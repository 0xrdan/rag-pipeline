/**
 * RAG Service - Retrieval-Augmented Generation
 *
 * Core RAG implementation that combines vector search with LLM generation.
 * Supports query expansion, hybrid search, and configurable prompts.
 *
 * @module rag
 */

import { VectorDatabase, RetrievedChunk, getVectorDB } from './vector-db';
import { EmbeddingService, getEmbeddingService } from './embeddings';

// =============================================================================
// Types
// =============================================================================

export interface RAGConfig {
  /** Number of chunks to retrieve (default: 5) */
  topK?: number;
  /** Minimum similarity threshold (default: 0.5) */
  threshold?: number;
  /** Enable query expansion with synonyms (default: true) */
  enableQueryExpansion?: boolean;
  /** Enable hybrid search with keyword boost (default: true) */
  enableHybridSearch?: boolean;
  /** Custom synonym map for query expansion */
  synonymMap?: Record<string, string[]>;
  /** System prompt for generation */
  systemPrompt?: string;
}

export interface RAGQuery {
  question: string;
  filters?: Record<string, any>;
  context?: string;
  conversationHistory?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

export interface RAGResult {
  /** Retrieved context chunks */
  chunks: RetrievedChunk[];
  /** Formatted context string for LLM */
  context: string;
  /** Query statistics */
  stats: {
    chunksRetrieved: number;
    avgSimilarity: number;
    queryTimeMs: number;
    expandedQuery?: string;
  };
}

export interface RAGResponse {
  answer: string;
  sources: Array<{
    title: string;
    url?: string;
    excerpt: string;
    score: number;
  }>;
  confidence: number;
  stats: RAGResult['stats'];
}

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_SYNONYMS: Record<string, string[]> = {
  'ml': ['ML', 'machine learning', 'Machine Learning'],
  'machine learning': ['ML', 'machine learning', 'Machine Learning'],
  'ai': ['AI', 'artificial intelligence', 'Artificial Intelligence'],
  'artificial intelligence': ['AI', 'artificial intelligence'],
  'dl': ['DL', 'deep learning', 'Deep Learning'],
  'deep learning': ['DL', 'deep learning', 'Deep Learning'],
  'nlp': ['NLP', 'natural language processing'],
  'natural language processing': ['NLP', 'natural language processing'],
  'llm': ['LLM', 'large language model', 'Large Language Model'],
  'large language model': ['LLM', 'large language model'],
  'rag': ['RAG', 'retrieval augmented generation'],
  'retrieval augmented generation': ['RAG', 'retrieval augmented generation'],
  'db': ['DB', 'database', 'Database'],
  'database': ['DB', 'database', 'Database'],
  'api': ['API', 'REST API', 'application programming interface'],
};

const DEFAULT_SYSTEM_PROMPT = `You are a helpful AI assistant. Answer questions based on the provided context.

Instructions:
- Base your answers on the provided context
- If the context doesn't contain enough information, say so
- Be concise but thorough
- Cite sources when relevant

Context:
{context}

Question: {question}`;

// =============================================================================
// RAG Service
// =============================================================================

export class RAGService {
  private vectorDB: VectorDatabase | null = null;
  private embeddingService: EmbeddingService | null = null;
  private config: Required<RAGConfig>;

  constructor(config?: RAGConfig) {
    this.config = {
      topK: config?.topK || 5,
      threshold: config?.threshold || 0.5,
      enableQueryExpansion: config?.enableQueryExpansion ?? true,
      enableHybridSearch: config?.enableHybridSearch ?? true,
      synonymMap: config?.synonymMap || DEFAULT_SYNONYMS,
      systemPrompt: config?.systemPrompt || DEFAULT_SYSTEM_PROMPT,
    };
  }

  /**
   * Initialize the RAG service
   */
  async initialize(): Promise<void> {
    this.vectorDB = await getVectorDB();
    this.embeddingService = getEmbeddingService();
    console.log('[RAG] Service initialized');
  }

  /**
   * Ensure service is initialized
   */
  private ensureInitialized(): void {
    if (!this.vectorDB || !this.embeddingService) {
      throw new Error('RAG service not initialized. Call initialize() first.');
    }
  }

  /**
   * Expand query with synonyms for better recall
   */
  private expandQuery(query: string): string {
    if (!this.config.enableQueryExpansion) {
      return query;
    }

    let expandedQuery = query.toLowerCase();

    for (const [key, synonyms] of Object.entries(this.config.synonymMap)) {
      const regex = new RegExp(`\\b${key}\\b`, 'gi');
      if (regex.test(expandedQuery)) {
        expandedQuery = expandedQuery + ' ' + synonyms.join(' ');
      }
    }

    return expandedQuery;
  }

  /**
   * Calculate keyword boost for hybrid search
   */
  private calculateKeywordBoost(query: string, chunk: RetrievedChunk): number {
    if (!this.config.enableHybridSearch) {
      return 0;
    }

    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 3);
    const contentLower = chunk.content.toLowerCase();
    const titleLower = chunk.metadata.title.toLowerCase();

    let boost = 0;

    // Exact phrase match gets biggest boost
    if (contentLower.includes(query.toLowerCase())) {
      boost += 0.2;
    }

    // Title match gets good boost
    const titleMatches = queryTerms.filter(term => titleLower.includes(term)).length;
    boost += (titleMatches / Math.max(queryTerms.length, 1)) * 0.15;

    // Content keyword matches get smaller boost
    const contentMatches = queryTerms.filter(term => contentLower.includes(term)).length;
    boost += (contentMatches / Math.max(queryTerms.length, 1)) * 0.1;

    return boost;
  }

  /**
   * Retrieve relevant chunks for a query
   */
  async retrieve(query: RAGQuery): Promise<RAGResult> {
    this.ensureInitialized();
    const startTime = Date.now();

    // 1. Expand query with synonyms
    const expandedQuery = this.expandQuery(query.question);

    // 2. Generate embedding
    const queryEmbedding = await this.embeddingService!.embed(expandedQuery);

    // 3. Similarity search
    let chunks = await this.vectorDB!.similaritySearch(
      queryEmbedding,
      this.config.topK,
      query.filters
    );

    // 4. Apply hybrid search boost
    if (this.config.enableHybridSearch) {
      chunks = chunks.map(chunk => ({
        ...chunk,
        score: Math.min(1.0, chunk.score + this.calculateKeywordBoost(query.question, chunk)),
      }));
      chunks.sort((a, b) => b.score - a.score);
    }

    // 5. Filter by threshold
    const filteredChunks = chunks.filter(chunk => chunk.score >= this.config.threshold);

    // 6. Build context
    const context = this.buildContext(filteredChunks);

    // 7. Calculate stats
    const avgSimilarity = filteredChunks.length > 0
      ? filteredChunks.reduce((sum, c) => sum + c.score, 0) / filteredChunks.length
      : 0;

    return {
      chunks: filteredChunks,
      context,
      stats: {
        chunksRetrieved: filteredChunks.length,
        avgSimilarity: Math.round(avgSimilarity * 100) / 100,
        queryTimeMs: Date.now() - startTime,
        expandedQuery: this.config.enableQueryExpansion ? expandedQuery : undefined,
      },
    };
  }

  /**
   * Build context string from chunks
   */
  private buildContext(chunks: RetrievedChunk[]): string {
    if (chunks.length === 0) {
      return 'No relevant context found.';
    }

    return chunks
      .sort((a, b) => b.score - a.score)
      .map((chunk, idx) => {
        const sourceInfo = [
          `[Source ${idx + 1}: ${chunk.metadata.title}]`,
          chunk.metadata.category ? `(${chunk.metadata.category})` : '',
          chunk.metadata.url ? `[${chunk.metadata.url}]` : '',
        ].filter(Boolean).join(' ');

        return `${sourceInfo}\n${chunk.content}`;
      })
      .join('\n\n---\n\n');
  }

  /**
   * Build the full prompt for generation
   */
  buildPrompt(question: string, context: string): string {
    return this.config.systemPrompt
      .replace('{context}', context)
      .replace('{question}', question);
  }

  /**
   * Calculate confidence score
   */
  calculateConfidence(chunks: RetrievedChunk[]): number {
    if (chunks.length === 0) return 0;

    const avgScore = chunks.reduce((sum, c) => sum + c.score, 0) / chunks.length;

    // Boost confidence if we have multiple high-quality chunks
    let confidence = avgScore;
    if (chunks.length >= 3 && avgScore > 0.8) {
      confidence = Math.min(0.95, avgScore * 1.1);
    }

    return Math.round(confidence * 100);
  }

  /**
   * Format sources for response
   */
  formatSources(chunks: RetrievedChunk[]): RAGResponse['sources'] {
    return chunks.map(chunk => ({
      title: chunk.metadata.title,
      url: chunk.metadata.url,
      excerpt: chunk.content.substring(0, 150) + (chunk.content.length > 150 ? '...' : ''),
      score: Math.round(chunk.score * 100) / 100,
    }));
  }

  /**
   * Get the system prompt
   */
  getSystemPrompt(): string {
    return this.config.systemPrompt;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<RAGConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<RAGConfig> {
    return { ...this.config };
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

let ragServiceInstance: RAGService | null = null;

export async function getRAGService(config?: RAGConfig): Promise<RAGService> {
  if (!ragServiceInstance) {
    ragServiceInstance = new RAGService(config);
    await ragServiceInstance.initialize();
  }
  return ragServiceInstance;
}

export function createRAGService(config?: RAGConfig): RAGService {
  return new RAGService(config);
}

export default RAGService;
