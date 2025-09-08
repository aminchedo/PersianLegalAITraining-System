// Document types
export interface Document {
  id: number;
  url: string;
  title: string;
  content: string;
  source: string;
  category?: string;
  scraped_at: string;
  content_hash: string;
  metadata?: string;
}

export interface SearchResult {
  documents: Document[];
  total: number;
}

// Scraping types
export interface ScrapingStatus {
  is_running: boolean;
  current_site: string | null;
  documents_scraped: number;
  errors: string[];
  started_at: string | null;
  estimated_completion: string | null;
}

// Classification types
export interface ClassificationResult {
  category: string;
  confidence: number;
  categories: {
    [key: string]: number;
  };
}

// Legal document categories
export enum DocumentCategory {
  CIVIL_LAW = 'civil_law',
  CRIMINAL_LAW = 'criminal_law',
  COMMERCIAL_LAW = 'commercial_law',
  ADMINISTRATIVE_LAW = 'administrative_law',
  CONSTITUTIONAL_LAW = 'constitutional_law',
  LABOR_LAW = 'labor_law',
  FAMILY_LAW = 'family_law',
  PROPERTY_LAW = 'property_law',
  TAX_LAW = 'tax_law',
  INTERNATIONAL_LAW = 'international_law',
  OTHER = 'other',
}

// Iranian legal sources
export const IRANIAN_LEGAL_SOURCES = [
  'divan-edalat.ir',
  'majlis.ir',
  'president.ir',
  'judiciary.ir',
  'rrk.ir',
  'shora-gc.ir',
] as const;

export type IranianLegalSource = typeof IRANIAN_LEGAL_SOURCES[number];

// Search filters
export interface SearchFilters {
  query: string;
  category?: DocumentCategory;
  source?: IranianLegalSource;
  dateFrom?: string;
  dateTo?: string;
  limit?: number;
  offset?: number;
}

// Statistics
export interface DocumentStats {
  total_documents: number;
  documents_by_category: Record<string, number>;
  documents_by_source: Record<string, number>;
  recent_scraping_activity: {
    date: string;
    documents_added: number;
  }[];
}