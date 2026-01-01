/**
 * Embedding Service - OpenAI text embeddings
 *
 * Provides text embedding generation using OpenAI's embedding models
 * with support for batch processing and dimension reduction.
 *
 * @module embeddings
 */

import OpenAI from 'openai';

// =============================================================================
// Types
// =============================================================================

export interface EmbeddingModelConfig {
  id: string;
  nativeDimensions: number;
  supportsDimensionReduction: boolean;
  costPer1MTokens: number;
}

export interface EmbeddingServiceConfig {
  /** OpenAI API key (defaults to OPENAI_API_KEY env var) */
  apiKey?: string;
  /** Model to use (default: text-embedding-3-large) */
  model?: string;
  /** Target dimensions for dimension reduction (optional) */
  dimensions?: number;
}

// =============================================================================
// Model Configurations
// =============================================================================

export const EMBEDDING_MODELS: Record<string, EmbeddingModelConfig> = {
  'text-embedding-3-small': {
    id: 'text-embedding-3-small',
    nativeDimensions: 1536,
    supportsDimensionReduction: true,
    costPer1MTokens: 0.02,
  },
  'text-embedding-3-large': {
    id: 'text-embedding-3-large',
    nativeDimensions: 3072,
    supportsDimensionReduction: true,
    costPer1MTokens: 0.13,
  },
  'text-embedding-ada-002': {
    id: 'text-embedding-ada-002',
    nativeDimensions: 1536,
    supportsDimensionReduction: false,
    costPer1MTokens: 0.10,
  },
};

// =============================================================================
// Embedding Service
// =============================================================================

export class EmbeddingService {
  private openai: OpenAI;
  private model: string;
  private modelConfig: EmbeddingModelConfig;
  private targetDimensions: number | null;

  constructor(config?: EmbeddingServiceConfig) {
    const apiKey = config?.apiKey || process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OpenAI API key is required. Set OPENAI_API_KEY or pass apiKey in config.');
    }

    this.openai = new OpenAI({ apiKey });
    this.model = config?.model || process.env.EMBEDDING_MODEL || 'text-embedding-3-large';
    this.modelConfig = EMBEDDING_MODELS[this.model] || EMBEDDING_MODELS['text-embedding-3-large'];

    // Optional dimension reduction
    this.targetDimensions = config?.dimensions || (process.env.EMBEDDING_DIMENSIONS ? parseInt(process.env.EMBEDDING_DIMENSIONS, 10) : null);

    if (this.targetDimensions && this.targetDimensions > this.modelConfig.nativeDimensions) {
      console.warn(`[Embeddings] Target dimensions ${this.targetDimensions} exceeds native ${this.modelConfig.nativeDimensions}. Using native.`);
      this.targetDimensions = null;
    }

    console.log(`[Embeddings] Initialized: ${this.model}, dimensions: ${this.getDimensions()}`);
  }

  /**
   * Preprocess text before embedding
   */
  private preprocessText(text: string): string {
    let cleaned = text.trim().replace(/\s+/g, ' ');

    // Remove markdown formatting for cleaner embeddings
    cleaned = cleaned
      .replace(/#+\s/g, '')                        // Headers
      .replace(/\*\*(.+?)\*\*/g, '$1')            // Bold
      .replace(/\*(.+?)\*/g, '$1')                // Italic
      .replace(/`(.+?)`/g, '$1')                  // Inline code
      .replace(/```[\s\S]*?```/g, '')             // Code blocks
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')    // Links
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1');  // Images

    return cleaned;
  }

  /**
   * Generate embedding for a single text
   */
  async embed(text: string): Promise<number[]> {
    const cleanText = this.preprocessText(text);

    if (!cleanText || cleanText.length === 0) {
      throw new Error('Text is empty after preprocessing');
    }

    const params: OpenAI.EmbeddingCreateParams = {
      model: this.model,
      input: cleanText,
    };

    if (this.targetDimensions && this.modelConfig.supportsDimensionReduction) {
      params.dimensions = this.targetDimensions;
    }

    const response = await this.openai.embeddings.create(params);
    return response.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple texts in batch
   * OpenAI allows up to 2048 inputs per request
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    const cleanTexts = texts
      .map(text => this.preprocessText(text))
      .filter(text => text && text.length > 0);

    if (cleanTexts.length === 0) {
      throw new Error('All texts are empty after preprocessing');
    }

    const batchSize = 2048;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < cleanTexts.length; i += batchSize) {
      const batch = cleanTexts.slice(i, i + batchSize);

      const params: OpenAI.EmbeddingCreateParams = {
        model: this.model,
        input: batch,
      };

      if (this.targetDimensions && this.modelConfig.supportsDimensionReduction) {
        params.dimensions = this.targetDimensions;
      }

      const response = await this.openai.embeddings.create(params);
      const embeddings = response.data.map(item => item.embedding);
      allEmbeddings.push(...embeddings);
    }

    return allEmbeddings;
  }

  /**
   * Get the embedding model being used
   */
  getModel(): string {
    return this.model;
  }

  /**
   * Get the embedding dimensions (target or native)
   */
  getDimensions(): number {
    return this.targetDimensions || this.modelConfig.nativeDimensions;
  }

  /**
   * Get model configuration
   */
  getModelConfig(): EmbeddingModelConfig {
    return this.modelConfig;
  }

  /**
   * Estimate cost for embedding a given number of tokens
   */
  estimateCost(tokenCount: number): number {
    return (tokenCount / 1_000_000) * this.modelConfig.costPer1MTokens;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

let embeddingServiceInstance: EmbeddingService | null = null;

export function getEmbeddingService(config?: EmbeddingServiceConfig): EmbeddingService {
  if (!embeddingServiceInstance) {
    embeddingServiceInstance = new EmbeddingService(config);
  }
  return embeddingServiceInstance;
}

export function createEmbeddingService(config?: EmbeddingServiceConfig): EmbeddingService {
  return new EmbeddingService(config);
}

export default EmbeddingService;
