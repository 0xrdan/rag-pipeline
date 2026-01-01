/**
 * Chunking Service - Document chunking for RAG
 *
 * Provides utilities for splitting documents into semantically
 * meaningful chunks suitable for embedding and retrieval.
 *
 * @module chunking
 */

import { ContentChunk } from './vector-db';

// =============================================================================
// Types
// =============================================================================

export interface ChunkingConfig {
  /** Maximum words per chunk (default: 500) */
  maxWordsPerChunk?: number;
  /** Minimum words per chunk (default: 50) */
  minWordsPerChunk?: number;
  /** Overlap words between chunks (default: 50) */
  overlapWords?: number;
  /** Split on these patterns (default: paragraphs, then sentences) */
  splitPatterns?: RegExp[];
}

export interface DocumentInput {
  id: string;
  content: string;
  title: string;
  source: string;
  url?: string;
  category?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

// =============================================================================
// Text Processing Utilities
// =============================================================================

/**
 * Strip markdown and HTML from text
 */
export function stripMarkdown(text: string): string {
  if (!text) return '';

  return text
    .replace(/#+\s/g, '')                        // Headers
    .replace(/\*\*(.+?)\*\*/g, '$1')            // Bold
    .replace(/\*(.+?)\*/g, '$1')                // Italic
    .replace(/`(.+?)`/g, '$1')                  // Inline code
    .replace(/```[\s\S]*?```/g, '')             // Code blocks
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')    // Links
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '')    // Images
    .replace(/<[^>]+>/g, '')                    // HTML tags
    .replace(/\n+/g, ' ')                       // Multiple newlines
    .replace(/\s+/g, ' ')                       // Multiple spaces
    .trim();
}

/**
 * Truncate text to a maximum word count
 */
export function truncateWords(text: string, maxWords: number): string {
  const words = text.split(/\s+/);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(' ') + '...';
}

/**
 * Count words in text
 */
export function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(w => w.length > 0).length;
}

// =============================================================================
// Chunking Service
// =============================================================================

export class ChunkingService {
  private config: Required<ChunkingConfig>;

  constructor(config?: ChunkingConfig) {
    this.config = {
      maxWordsPerChunk: config?.maxWordsPerChunk || 500,
      minWordsPerChunk: config?.minWordsPerChunk || 50,
      overlapWords: config?.overlapWords || 50,
      splitPatterns: config?.splitPatterns || [
        /\n\n+/,       // Double newlines (paragraphs)
        /\n/,          // Single newlines
        /(?<=[.!?])\s+/, // Sentence boundaries
      ],
    };
  }

  /**
   * Chunk a document into semantic pieces
   */
  chunkDocument(doc: DocumentInput): ContentChunk[] {
    const chunks: ContentChunk[] = [];
    const cleanContent = stripMarkdown(doc.content);

    if (countWords(cleanContent) < this.config.minWordsPerChunk) {
      // Document is small enough to be a single chunk
      chunks.push(this.createChunk(doc, cleanContent, 0, 1));
      return chunks;
    }

    // Split into paragraphs first
    const paragraphs = cleanContent.split(/\n\n+/).filter(p => p.trim().length > 0);

    let currentChunk = '';
    let chunkIndex = 0;

    for (const paragraph of paragraphs) {
      const paragraphWords = countWords(paragraph);
      const currentWords = countWords(currentChunk);

      if (currentWords + paragraphWords <= this.config.maxWordsPerChunk) {
        // Add to current chunk
        currentChunk = currentChunk ? `${currentChunk}\n\n${paragraph}` : paragraph;
      } else if (paragraphWords > this.config.maxWordsPerChunk) {
        // Paragraph itself is too large, need to split it
        if (currentChunk) {
          chunks.push(this.createChunk(doc, currentChunk.trim(), chunkIndex++, -1));
          currentChunk = '';
        }

        // Split large paragraph by sentences
        const sentences = paragraph.split(/(?<=[.!?])\s+/);
        let sentenceChunk = '';

        for (const sentence of sentences) {
          const sentenceWords = countWords(sentence);
          const chunkWords = countWords(sentenceChunk);

          if (chunkWords + sentenceWords <= this.config.maxWordsPerChunk) {
            sentenceChunk = sentenceChunk ? `${sentenceChunk} ${sentence}` : sentence;
          } else {
            if (sentenceChunk) {
              chunks.push(this.createChunk(doc, sentenceChunk.trim(), chunkIndex++, -1));
            }
            sentenceChunk = sentence;
          }
        }

        if (sentenceChunk) {
          currentChunk = sentenceChunk;
        }
      } else {
        // Start new chunk
        if (currentChunk) {
          chunks.push(this.createChunk(doc, currentChunk.trim(), chunkIndex++, -1));
        }
        currentChunk = paragraph;
      }
    }

    // Don't forget the last chunk
    if (currentChunk && countWords(currentChunk) >= this.config.minWordsPerChunk) {
      chunks.push(this.createChunk(doc, currentChunk.trim(), chunkIndex++, -1));
    } else if (currentChunk && chunks.length > 0) {
      // Append small remaining content to last chunk
      const lastChunk = chunks[chunks.length - 1];
      lastChunk.content = `${lastChunk.content}\n\n${currentChunk.trim()}`;
    } else if (currentChunk) {
      // Only chunk, even if small
      chunks.push(this.createChunk(doc, currentChunk.trim(), chunkIndex++, -1));
    }

    // Update total chunks count
    const totalChunks = chunks.length;
    chunks.forEach(chunk => {
      chunk.metadata.totalChunks = totalChunks;
    });

    return chunks;
  }

  /**
   * Chunk multiple documents
   */
  chunkDocuments(docs: DocumentInput[]): ContentChunk[] {
    const allChunks: ContentChunk[] = [];
    for (const doc of docs) {
      allChunks.push(...this.chunkDocument(doc));
    }
    return allChunks;
  }

  /**
   * Create a chunk with proper metadata
   */
  private createChunk(
    doc: DocumentInput,
    content: string,
    chunkIndex: number,
    totalChunks: number
  ): ContentChunk {
    const chunkTitle = totalChunks === 1
      ? doc.title
      : `${doc.title} (Part ${chunkIndex + 1})`;

    return {
      id: `${doc.source}_${doc.id}_chunk_${chunkIndex}`,
      content,
      metadata: {
        source: doc.source,
        sourceId: doc.id,
        title: chunkTitle,
        category: doc.category,
        tags: doc.tags,
        url: doc.url,
        chunkIndex,
        totalChunks,
        ...doc.metadata,
      },
    };
  }

  /**
   * Chunk by sections (using ## headers)
   */
  chunkBySections(doc: DocumentInput): ContentChunk[] {
    const chunks: ContentChunk[] = [];
    const sections = doc.content.split(/^## /gm);

    // First section is before any ## header
    if (sections[0] && sections[0].trim().length > this.config.minWordsPerChunk) {
      const introContent = stripMarkdown(sections[0]);
      chunks.push(this.createChunk(
        { ...doc, title: `${doc.title} - Introduction` },
        truncateWords(introContent, this.config.maxWordsPerChunk),
        0,
        sections.length
      ));
    }

    // Process each section
    for (let i = 1; i < sections.length; i++) {
      const section = sections[i];
      if (!section || section.trim().length < 30) continue;

      const lines = section.split('\n');
      const sectionTitle = lines[0]?.trim() || `Section ${i}`;
      const sectionContent = stripMarkdown(lines.slice(1).join('\n'));

      if (sectionContent.length < 30) continue;

      chunks.push(this.createChunk(
        { ...doc, title: `${doc.title} - ${sectionTitle}` },
        truncateWords(sectionContent, this.config.maxWordsPerChunk),
        i,
        sections.length
      ));
    }

    return chunks;
  }

  /**
   * Create an overview chunk (title + summary)
   */
  createOverviewChunk(
    doc: DocumentInput,
    fields: Array<{ label?: string; value: string | undefined }>
  ): ContentChunk {
    const content = fields
      .filter(f => f.value)
      .map(f => f.label ? `${f.label}: ${f.value}` : f.value)
      .join('. ');

    return this.createChunk(doc, content, 0, 1);
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

let chunkingServiceInstance: ChunkingService | null = null;

export function getChunkingService(config?: ChunkingConfig): ChunkingService {
  if (!chunkingServiceInstance) {
    chunkingServiceInstance = new ChunkingService(config);
  }
  return chunkingServiceInstance;
}

export function createChunkingService(config?: ChunkingConfig): ChunkingService {
  return new ChunkingService(config);
}

export default ChunkingService;
