"""
Persian Legal Data Processing Pipeline
پایپ‌لاین پردازش داده‌های حقوقی فارسی
"""

import asyncio
import aiohttp
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import re
import hashlib
from dataclasses import dataclass
import pandas as pd
from bs4 import BeautifulSoup
import hazm
from transformers import AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    naab_api_key: Optional[str] = None
    naab_base_url: str = "https://api.naab.ir/v1"
    qavanin_base_url: str = "https://qavanin.ir"
    majles_base_url: str = "https://majlis.ir"
    max_documents_per_source: int = 1000
    request_delay: float = 2.0
    timeout: int = 30

@dataclass
class DocumentQuality:
    """Document quality metrics"""
    score: float
    length: int
    language_confidence: float
    legal_relevance: float
    completeness: float
    issues: List[str]

class NaabCorpusConnector:
    """
    Real connection to Naab Corpus API for Persian legal documents
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.naab.ir/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.rate_limit_delay = 2.0
        
    async def connect(self):
        """Initialize API connection"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'PersianLegalAI/1.0'
                }
            )
            logger.info("Connected to Naab Corpus API")
            
        except Exception as e:
            logger.error(f"Failed to connect to Naab Corpus API: {e}")
            raise
    
    async def fetch_legal_documents(self, filters: Dict[str, Any], limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch legal documents from Naab Corpus"""
        try:
            if not self.session:
                await self.connect()
            
            documents = []
            page = 1
            page_size = min(100, limit)
            
            while len(documents) < limit:
                # Build query parameters
                params = {
                    'page': page,
                    'size': page_size,
                    **filters
                }
                
                # Make API request
                async with self.session.get(f"{self.base_url}/documents", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not data.get('documents'):
                            break
                        
                        documents.extend(data['documents'])
                        page += 1
                        
                        # Rate limiting
                        await asyncio.sleep(self.rate_limit_delay)
                        
                    elif response.status == 429:  # Rate limited
                        logger.warning("Rate limited, waiting longer...")
                        await asyncio.sleep(5.0)
                        
                    else:
                        logger.error(f"API request failed: {response.status}")
                        break
            
            logger.info(f"Fetched {len(documents)} documents from Naab Corpus")
            return documents[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch documents from Naab Corpus: {e}")
            return []
    
    async def get_document_quality_score(self, document: Dict[str, Any]) -> float:
        """Assess document quality"""
        try:
            # Extract text content
            content = document.get('content', '')
            title = document.get('title', '')
            
            # Quality metrics
            length_score = min(len(content) / 1000, 1.0)  # Prefer longer documents
            title_score = 1.0 if title else 0.5
            
            # Language confidence (basic Persian text detection)
            persian_chars = len(re.findall(r'[\u0600-\u06FF]', content))
            total_chars = len(content)
            language_score = persian_chars / total_chars if total_chars > 0 else 0.0
            
            # Legal relevance (keyword matching)
            legal_keywords = [
                'قانون', 'ماده', 'بند', 'تبصره', 'مقررات', 'دستورالعمل',
                'آیین\u200cنامه', 'بخشنامه', 'رای', 'حکم', 'دادگاه', 'قاضی'
            ]
            legal_score = sum(1 for keyword in legal_keywords if keyword in content) / len(legal_keywords)
            
            # Overall quality score
            quality_score = (length_score * 0.3 + title_score * 0.2 + 
                           language_score * 0.3 + legal_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Failed to assess document quality: {e}")
            return 0.0
    
    async def close(self):
        """Close API connection"""
        if self.session:
            await self.session.close()

class QavaninPortalScraper:
    """
    Real web scraping for Qavanin.ir portal
    """
    
    def __init__(self, base_url: str = "https://qavanin.ir"):
        self.base_url = base_url
        self.session = None
        self.delay_between_requests = 2.0
        
    async def connect(self):
        """Initialize scraping session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            logger.info("Connected to Qavanin portal")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qavanin portal: {e}")
            raise
    
    async def scrape_laws_by_category(self, category: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Scrape laws by category"""
        try:
            if not self.session:
                await self.connect()
            
            laws = []
            
            for page in range(1, max_pages + 1):
                # Build URL for category page
                url = f"{self.base_url}/category/{category}?page={page}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract law links
                        law_links = soup.find_all('a', href=re.compile(r'/law/'))
                        
                        for link in law_links:
                            law_url = link.get('href')
                            if law_url.startswith('/'):
                                law_url = self.base_url + law_url
                            
                            # Extract law content
                            law_data = await self.extract_law_content(law_url)
                            if law_data:
                                laws.append(law_data)
                        
                        # Rate limiting
                        await asyncio.sleep(self.delay_between_requests)
                        
                    else:
                        logger.warning(f"Failed to fetch page {page}: {response.status}")
                        break
            
            logger.info(f"Scraped {len(laws)} laws from category {category}")
            return laws
            
        except Exception as e:
            logger.error(f"Failed to scrape laws by category: {e}")
            return []
    
    async def extract_law_content(self, law_url: str) -> Optional[Dict[str, Any]]:
        """Extract content from individual law page"""
        try:
            async with self.session.get(law_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.find('h1') or soup.find('title')
                    title = title_elem.get_text().strip() if title_elem else "Unknown Title"
                    
                    # Extract content
                    content_elem = soup.find('div', class_='content') or soup.find('article')
                    if not content_elem:
                        content_elem = soup.find('body')
                    
                    content = content_elem.get_text().strip() if content_elem else ""
                    
                    # Extract metadata
                    metadata = {
                        'url': law_url,
                        'scraped_at': datetime.now().isoformat(),
                        'source': 'qavanin.ir'
                    }
                    
                    return {
                        'title': title,
                        'content': content,
                        'metadata': metadata
                    }
                
        except Exception as e:
            logger.error(f"Failed to extract law content from {law_url}: {e}")
            return None
    
    async def close(self):
        """Close scraping session"""
        if self.session:
            await self.session.close()

class MajlesDataExtractor:
    """
    Extract data from Majles (Parliament) sources
    """
    
    def __init__(self, base_url: str = "https://majlis.ir"):
        self.base_url = base_url
        self.session = None
        
    async def connect(self):
        """Initialize extraction session"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            logger.info("Connected to Majles data source")
            
        except Exception as e:
            logger.error(f"Failed to connect to Majles: {e}")
            raise
    
    async def extract_parliamentary_documents(self, date_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Extract parliamentary documents from date range"""
        try:
            if not self.session:
                await self.connect()
            
            documents = []
            start_date, end_date = date_range
            
            # Extract documents from date range
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Build URL for specific date
                url = f"{self.base_url}/archive/{date_str}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract document links
                        doc_links = soup.find_all('a', href=re.compile(r'/document/'))
                        
                        for link in doc_links:
                            doc_url = link.get('href')
                            if doc_url.startswith('/'):
                                doc_url = self.base_url + doc_url
                            
                            # Extract document content
                            doc_data = await self.extract_document_content(doc_url)
                            if doc_data:
                                documents.append(doc_data)
                
                current_date += timedelta(days=1)
                await asyncio.sleep(1.0)  # Rate limiting
            
            logger.info(f"Extracted {len(documents)} parliamentary documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract parliamentary documents: {e}")
            return []
    
    async def extract_document_content(self, doc_url: str) -> Optional[Dict[str, Any]]:
        """Extract content from individual document"""
        try:
            async with self.session.get(doc_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title and content
                    title = soup.find('h1').get_text().strip() if soup.find('h1') else "Unknown Title"
                    content = soup.find('div', class_='content').get_text().strip() if soup.find('div', class_='content') else ""
                    
                    return {
                        'title': title,
                        'content': content,
                        'metadata': {
                            'url': doc_url,
                            'source': 'majlis.ir',
                            'extracted_at': datetime.now().isoformat()
                        }
                    }
                
        except Exception as e:
            logger.error(f"Failed to extract document content: {e}")
            return None
    
    async def close(self):
        """Close extraction session"""
        if self.session:
            await self.session.close()

class PersianLegalDataProcessor:
    """
    Main data processing pipeline for Persian legal documents
    """
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self.naab_connector = None
        self.qavanin_scraper = None
        self.majles_extractor = None
        self.tokenizer = None
        self.normalizer = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize data processing components"""
        try:
            # Initialize Naab connector if API key is available
            if self.config.naab_api_key:
                self.naab_connector = NaabCorpusConnector(
                    self.config.naab_api_key,
                    self.config.naab_base_url
                )
            
            # Initialize scrapers
            self.qavanin_scraper = QavaninPortalScraper(self.config.qavanin_base_url)
            self.majles_extractor = MajlesDataExtractor(self.config.majles_base_url)
            
            # Initialize Persian text processing
            self.normalizer = hazm.Normalizer()
            
            # Initialize tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
            except Exception as e:
                logger.warning(f"Could not load Persian tokenizer: {e}")
                self.tokenizer = None
            
            logger.info("Data processing components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    async def fetch_legal_documents(self, sources: List[str], date_range: Optional[Tuple[datetime, datetime]] = None,
                                  categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch legal documents from multiple sources"""
        try:
            all_documents = []
            
            # Fetch from Naab Corpus
            if 'naab' in sources and self.naab_connector:
                await self.naab_connector.connect()
                naab_docs = await self.naab_connector.fetch_legal_documents(
                    filters={'categories': categories} if categories else {},
                    limit=self.config.max_documents_per_source
                )
                all_documents.extend(naab_docs)
            
            # Fetch from Qavanin portal
            if 'qavanin' in sources:
                await self.qavanin_scraper.connect()
                for category in (categories or ['laws', 'regulations']):
                    qavanin_docs = await self.qavanin_scraper.scrape_laws_by_category(
                        category, max_pages=10
                    )
                    all_documents.extend(qavanin_docs)
            
            # Fetch from Majles
            if 'majles' in sources and date_range:
                await self.majles_extractor.connect()
                majles_docs = await self.majles_extractor.extract_parliamentary_documents(date_range)
                all_documents.extend(majles_docs)
            
            logger.info(f"Fetched {len(all_documents)} documents from {len(sources)} sources")
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to fetch legal documents: {e}")
            return []
    
    def preprocess_persian_text(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess Persian text using Hazm"""
        try:
            processed_docs = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Normalize Persian text
                normalized_content = self.normalizer.normalize(content)
                normalized_title = self.normalizer.normalize(title)
                
                # Tokenize if tokenizer is available
                tokens = None
                if self.tokenizer:
                    tokens = self.tokenizer.tokenize(normalized_content)
                
                processed_doc = {
                    **doc,
                    'content': normalized_content,
                    'title': normalized_title,
                    'tokens': tokens,
                    'word_count': len(normalized_content.split()),
                    'char_count': len(normalized_content),
                    'processed_at': datetime.now().isoformat()
                }
                
                processed_docs.append(processed_doc)
            
            logger.info(f"Preprocessed {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to preprocess Persian text: {e}")
            return documents
    
    def assess_document_quality(self, documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], DocumentQuality]]:
        """Assess quality of documents"""
        try:
            quality_assessments = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Calculate quality metrics
                length = len(content)
                length_score = min(length / 1000, 1.0)
                
                # Language confidence
                persian_chars = len(re.findall(r'[\u0600-\u06FF]', content))
                total_chars = len(content)
                language_confidence = persian_chars / total_chars if total_chars > 0 else 0.0
                
                # Legal relevance
                legal_keywords = [
                    'قانون', 'ماده', 'بند', 'تبصره', 'مقررات', 'دستورالعمل',
                    'آیین\u200cنامه', 'بخشنامه', 'رای', 'حکم', 'دادگاه', 'قاضی',
                    'حقوق', 'قضایی', 'قانونی', 'مبانی', 'اصول'
                ]
                legal_relevance = sum(1 for keyword in legal_keywords if keyword in content) / len(legal_keywords)
                
                # Completeness
                completeness = 1.0 if title and content else 0.5
                
                # Identify issues
                issues = []
                if length < 100:
                    issues.append("Too short")
                if language_confidence < 0.5:
                    issues.append("Low Persian content")
                if legal_relevance < 0.1:
                    issues.append("Low legal relevance")
                if not title:
                    issues.append("Missing title")
                
                # Overall quality score
                quality_score = (length_score * 0.3 + language_confidence * 0.3 + 
                               legal_relevance * 0.3 + completeness * 0.1)
                
                quality = DocumentQuality(
                    score=quality_score,
                    length=length,
                    language_confidence=language_confidence,
                    legal_relevance=legal_relevance,
                    completeness=completeness,
                    issues=issues
                )
                
                quality_assessments.append((doc, quality))
            
            logger.info(f"Assessed quality of {len(quality_assessments)} documents")
            return quality_assessments
            
        except Exception as e:
            logger.error(f"Failed to assess document quality: {e}")
            return [(doc, DocumentQuality(score=0.0, length=0, language_confidence=0.0, 
                                        legal_relevance=0.0, completeness=0.0, issues=["Assessment failed"])) 
                   for doc in documents]
    
    def filter_high_quality_documents(self, quality_assessments: List[Tuple[Dict[str, Any], DocumentQuality]], 
                                    min_quality_score: float = 0.6) -> List[Dict[str, Any]]:
        """Filter documents by quality score"""
        try:
            high_quality_docs = []
            
            for doc, quality in quality_assessments:
                if quality.score >= min_quality_score:
                    # Add quality metrics to document
                    doc['quality_metrics'] = {
                        'score': quality.score,
                        'length': quality.length,
                        'language_confidence': quality.language_confidence,
                        'legal_relevance': quality.legal_relevance,
                        'completeness': quality.completeness,
                        'issues': quality.issues
                    }
                    high_quality_docs.append(doc)
            
            logger.info(f"Filtered {len(high_quality_docs)} high-quality documents from {len(quality_assessments)} total")
            return high_quality_docs
            
        except Exception as e:
            logger.error(f"Failed to filter high-quality documents: {e}")
            return []
    
    def create_training_datasets(self, documents: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Create task-specific training datasets"""
        try:
            if task_type == 'question_answering':
                return self._create_qa_dataset(documents)
            elif task_type == 'named_entity_recognition':
                return self._create_ner_dataset(documents)
            elif task_type == 'text_classification':
                return self._create_classification_dataset(documents)
            elif task_type == 'text_generation':
                return self._create_generation_dataset(documents)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            return {}
    
    def _create_qa_dataset(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create question-answering dataset"""
        try:
            qa_pairs = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Generate questions from legal text
                questions = self._generate_legal_questions(content, title)
                
                for question, answer in questions:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'context': content[:500],  # Truncate context
                        'source': doc.get('metadata', {}).get('source', 'unknown'),
                        'document_id': doc.get('id', hashlib.md5(content.encode()).hexdigest())
                    })
            
            return {
                'task_type': 'question_answering',
                'dataset': qa_pairs,
                'size': len(qa_pairs),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create QA dataset: {e}")
            return {}
    
    def _generate_legal_questions(self, content: str, title: str) -> List[Tuple[str, str]]:
        """Generate legal questions from content"""
        try:
            questions = []
            
            # Extract legal articles/sections
            articles = re.findall(r'ماده\s+(\d+)[:.]?\s*(.+?)(?=ماده|\Z)', content, re.DOTALL)
            
            for article_num, article_text in articles[:5]:  # Limit to first 5 articles
                if len(article_text.strip()) > 50:
                    question = f"ماده {article_num} {title} چه می\u200cگوید؟"
                    answer = article_text.strip()[:200]  # Truncate answer
                    questions.append((question, answer))
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to generate legal questions: {e}")
            return []
    
    def _create_ner_dataset(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create named entity recognition dataset"""
        try:
            ner_data = []
            
            for doc in documents:
                content = doc.get('content', '')
                
                # Extract legal entities
                entities = self._extract_legal_entities(content)
                
                if entities:
                    ner_data.append({
                        'text': content,
                        'entities': entities,
                        'source': doc.get('metadata', {}).get('source', 'unknown')
                    })
            
            return {
                'task_type': 'named_entity_recognition',
                'dataset': ner_data,
                'size': len(ner_data),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create NER dataset: {e}")
            return {}
    
    def _extract_legal_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract legal entities from text"""
        try:
            entities = []
            
            # Extract law numbers
            law_numbers = re.findall(r'قانون\s+(\d+)', content)
            for num in law_numbers:
                entities.append({
                    'text': f"قانون {num}",
                    'label': 'LAW_NUMBER',
                    'start': content.find(f"قانون {num}"),
                    'end': content.find(f"قانون {num}") + len(f"قانون {num}")
                })
            
            # Extract article numbers
            articles = re.findall(r'ماده\s+(\d+)', content)
            for num in articles:
                entities.append({
                    'text': f"ماده {num}",
                    'label': 'ARTICLE_NUMBER',
                    'start': content.find(f"ماده {num}"),
                    'end': content.find(f"ماده {num}") + len(f"ماده {num}")
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract legal entities: {e}")
            return []
    
    def _create_classification_dataset(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create text classification dataset"""
        try:
            classification_data = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Classify document type
                doc_type = self._classify_document_type(content, title)
                
                classification_data.append({
                    'text': content,
                    'label': doc_type,
                    'title': title,
                    'source': doc.get('metadata', {}).get('source', 'unknown')
                })
            
            return {
                'task_type': 'text_classification',
                'dataset': classification_data,
                'size': len(classification_data),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create classification dataset: {e}")
            return {}
    
    def _classify_document_type(self, content: str, title: str) -> str:
        """Classify document type"""
        try:
            # Simple rule-based classification
            if 'قانون' in title or 'قانون' in content:
                return 'law'
            elif 'آیین\u200cنامه' in title or 'آیین\u200cنامه' in content:
                return 'regulation'
            elif 'دستورالعمل' in title or 'دستورالعمل' in content:
                return 'instruction'
            elif 'رای' in title or 'حکم' in content:
                return 'judgment'
            else:
                return 'other'
                
        except Exception as e:
            logger.error(f"Failed to classify document type: {e}")
            return 'unknown'
    
    def _create_generation_dataset(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create text generation dataset"""
        try:
            generation_data = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Create generation prompts
                prompts = [
                    f"خلاصه‌ای از {title} ارائه دهید:",
                    f"مبانی حقوقی {title} چیست؟",
                    f"توضیح دهید که {title} چگونه عمل می\u200cکند:"
                ]
                
                for prompt in prompts:
                    generation_data.append({
                        'prompt': prompt,
                        'completion': content[:500],  # Truncate completion
                        'title': title,
                        'source': doc.get('metadata', {}).get('source', 'unknown')
                    })
            
            return {
                'task_type': 'text_generation',
                'dataset': generation_data,
                'size': len(generation_data),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create generation dataset: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.naab_connector:
                await self.naab_connector.close()
            if self.qavanin_scraper:
                await self.qavanin_scraper.close()
            if self.majles_extractor:
                await self.majles_extractor.close()
            
            logger.info("Data processor cleanup complete")
            
        except Exception as e:
            logger.error(f"Failed to cleanup data processor: {e}")