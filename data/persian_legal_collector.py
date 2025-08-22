"""
Persian Legal Data Collector
Comprehensive data collection from verified Persian legal sources with advanced processing
"""

import os
import re
import asyncio
import aiohttp
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import json
import gzip
import sqlite3
from datetime import datetime, timedelta

# Text processing
import spacy
import nltk
from hazm import Normalizer, Stemmer, Lemmatizer, POSTagger
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# Selenium for dynamic content
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Local imports
from config.training_config import DataCollectionConfig, get_config


@dataclass
class LegalDocument:
    """Represents a Persian legal document with metadata"""
    
    id: str
    title: str
    content: str
    source: str
    url: str
    document_type: str
    date_published: Optional[str] = None
    law_category: Optional[str] = None
    legal_references: List[str] = None
    entities: Dict[str, List[str]] = None
    quality_score: float = 0.0
    language: str = "persian"
    word_count: int = 0
    processed_at: str = None
    
    def __post_init__(self):
        if self.legal_references is None:
            self.legal_references = []
        if self.entities is None:
            self.entities = {}
        if self.processed_at is None:
            self.processed_at = datetime.now().isoformat()
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class PersianTextProcessor:
    """Advanced Persian text processing with legal domain specialization"""
    
    def __init__(self):
        # Initialize Persian NLP tools
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()
        self.pos_tagger = POSTagger(model='resources/postagger.model')
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual model
            logger.info("SpaCy multilingual model loaded")
        except OSError:
            self.nlp = None
            logger.warning("SpaCy model not available - using basic processing")
        
        # Persian legal patterns
        self._initialize_legal_patterns()
        
        # Quality assessment criteria
        self.quality_criteria = {
            'min_words': 50,
            'max_words': 50000,
            'min_sentences': 3,
            'legal_keyword_threshold': 0.02,
            'persian_text_ratio': 0.8
        }
        
        logger.info("Persian text processor initialized")
    
    def _initialize_legal_patterns(self) -> None:
        """Initialize Persian legal text patterns"""
        
        # Persian law article patterns
        self.law_patterns = {
            'article': re.compile(r'ماده\s*(\d+)[\s\u200c]*[-–]\s*(.+?)(?=ماده|\Z)', re.DOTALL),
            'law_reference': re.compile(r'قانون\s+(.+?)\s+مصوب\s+(\d{4})'),
            'court_reference': re.compile(r'(دادگاه|دیوان|شورای|کمیسیون)\s+(.+?)(?=\s|$)'),
            'date_persian': re.compile(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})'),
            'currency': re.compile(r'(\d+(?:,\d{3})*)\s*(?:ریال|تومان|درهم)'),
            'legal_entity': re.compile(r'(شرکت|سازمان|اداره|وزارت|بانک)\s+(.+?)(?=\s|$)'),
            'judgment': re.compile(r'(حکم|رأی|قرار|دستور)\s+(.+?)(?=\.|$)')
        }
        
        # Persian legal keywords
        self.legal_keywords = {
            'civil_law': ['قانون مدنی', 'حقوق مدنی', 'دعوای مدنی', 'قرارداد', 'تعهد'],
            'criminal_law': ['قانون جزا', 'جرم', 'مجازات', 'تعزیر', 'قصاص'],
            'commercial_law': ['قانون تجارت', 'شرکت', 'تجاری', 'بازرگانی', 'اقتصادی'],
            'constitutional_law': ['قانون اساسی', 'مجلس', 'شورای نگهبان', 'رئیس جمهور'],
            'administrative_law': ['قانون اداری', 'اداره', 'خدمات عمومی', 'مأمور دولتی']
        }
        
        # Combine all keywords
        self.all_legal_keywords = set()
        for category_keywords in self.legal_keywords.values():
            self.all_legal_keywords.update(category_keywords)
    
    def normalize_text(self, text: str) -> str:
        """Comprehensive Persian text normalization"""
        
        # Basic normalization
        text = self.normalizer.normalize(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Persian numbers
        persian_numbers = '۰۱۲۳۴۵۶۷۸۹'
        arabic_numbers = '0123456789'
        for p, a in zip(persian_numbers, arabic_numbers):
            text = text.replace(p, a)
        
        # Handle ZWNJ (Zero Width Non-Joiner)
        text = re.sub(r'\u200c+', '\u200c', text)
        
        return text.strip()
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from Persian text"""
        
        entities = {
            'articles': [],
            'laws': [],
            'courts': [],
            'dates': [],
            'organizations': [],
            'amounts': []
        }
        
        # Extract articles
        for match in self.law_patterns['article'].finditer(text):
            entities['articles'].append(f"ماده {match.group(1)}")
        
        # Extract law references
        for match in self.law_patterns['law_reference'].finditer(text):
            entities['laws'].append(f"{match.group(1)} ({match.group(2)})")
        
        # Extract court references
        for match in self.law_patterns['court_reference'].finditer(text):
            entities['courts'].append(f"{match.group(1)} {match.group(2)}")
        
        # Extract dates
        for match in self.law_patterns['date_persian'].finditer(text):
            entities['dates'].append(match.group(0))
        
        # Extract organizations
        for match in self.law_patterns['legal_entity'].finditer(text):
            entities['organizations'].append(f"{match.group(1)} {match.group(2)}")
        
        # Extract monetary amounts
        for match in self.law_patterns['currency'].finditer(text):
            entities['amounts'].append(match.group(0))
        
        # Use spaCy if available for additional entity extraction
        if self.nlp:
            doc = self.nlp(text[:1000])  # Limit text length for performance
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                    category = 'organizations' if ent.label_ == 'ORG' else 'entities'
                    if category not in entities:
                        entities[category] = []
                    entities[category].append(ent.text)
        
        # Remove duplicates and empty entries
        for key in entities:
            entities[key] = list(set([e for e in entities[key] if e.strip()]))
        
        return entities
    
    def assess_quality(self, text: str) -> float:
        """Assess the quality of Persian legal text"""
        
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 1.0
        
        # Word count check
        words = text.split()
        word_count = len(words)
        
        if word_count < self.quality_criteria['min_words']:
            score *= 0.5
        elif word_count > self.quality_criteria['max_words']:
            score *= 0.7
        
        # Sentence count check
        sentences = re.split(r'[.!?؟]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.quality_criteria['min_sentences']:
            score *= 0.6
        
        # Persian text ratio
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(text)
        persian_ratio = persian_chars / total_chars if total_chars > 0 else 0
        
        if persian_ratio < self.quality_criteria['persian_text_ratio']:
            score *= persian_ratio
        
        # Legal keyword density
        text_lower = text.lower()
        legal_keyword_count = sum(1 for keyword in self.all_legal_keywords if keyword in text_lower)
        keyword_density = legal_keyword_count / word_count if word_count > 0 else 0
        
        if keyword_density >= self.quality_criteria['legal_keyword_threshold']:
            score *= 1.2  # Boost for legal content
        else:
            score *= 0.8
        
        # Structural quality (presence of legal patterns)
        pattern_matches = 0
        for pattern in self.law_patterns.values():
            if pattern.search(text):
                pattern_matches += 1
        
        if pattern_matches > 0:
            score *= (1 + 0.1 * pattern_matches)
        
        return min(score, 1.0)
    
    def categorize_document(self, text: str) -> str:
        """Categorize legal document based on content"""
        
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.legal_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) == 0:
            return 'general_legal'
        
        return max(category_scores, key=category_scores.get)


class DataSourceConnector:
    """Handles connections to various Persian legal data sources"""
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.selenium_driver: Optional[webdriver.Chrome] = None
        
        # Request settings
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fa,en-US;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        logger.info("Data source connector initialized")
    
    async def initialize(self) -> None:
        """Initialize HTTP session and Selenium driver"""
        
        # Initialize aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self.headers
        )
        
        # Initialize Selenium driver if available
        if SELENIUM_AVAILABLE:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            try:
                self.selenium_driver = webdriver.Chrome(options=chrome_options)
                logger.info("Selenium driver initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Selenium driver: {e}")
                self.selenium_driver = None
        
        logger.info("Data source connector initialization completed")
    
    async def fetch_url(self, url: str, use_selenium: bool = False) -> Optional[str]:
        """Fetch content from URL with retry logic"""
        
        for attempt in range(self.config.retry_attempts):
            try:
                if use_selenium and self.selenium_driver:
                    return await self._fetch_with_selenium(url)
                else:
                    return await self._fetch_with_aiohttp(url)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.rate_limit_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch {url} after {self.config.retry_attempts} attempts")
                    return None
        
        return None
    
    async def _fetch_with_aiohttp(self, url: str) -> str:
        """Fetch content using aiohttp"""
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            content = await response.text(encoding='utf-8')
            return content
    
    async def _fetch_with_selenium(self, url: str) -> str:
        """Fetch content using Selenium for dynamic content"""
        
        self.selenium_driver.get(url)
        
        # Wait for content to load
        WebDriverWait(self.selenium_driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        await asyncio.sleep(2)
        
        return self.selenium_driver.page_source
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        
        if self.session:
            await self.session.close()
        
        if self.selenium_driver:
            self.selenium_driver.quit()
        
        logger.info("Data source connector cleanup completed")


class PersianLegalDataCollector:
    """Main Persian legal data collection system"""
    
    def __init__(
        self,
        cache_dir: Path = Path("./data/cache"),
        max_workers: int = 8,
        enable_caching: bool = True
    ):
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # Initialize components
        self.config = DataCollectionConfig()
        self.text_processor = PersianTextProcessor()
        self.connector = DataSourceConnector(self.config)
        
        # Data storage
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "legal_documents.db"
        
        # Collection state
        self.collection_active = False
        self.collected_documents = 0
        self.failed_urls = 0
        self.duplicate_count = 0
        
        # Statistics
        self.collection_stats = {
            'documents_by_source': {},
            'documents_by_category': {},
            'quality_distribution': [],
            'collection_timeline': []
        }
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Persian Legal Data Collector initialized with {max_workers} workers")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for document storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    source TEXT,
                    url TEXT,
                    document_type TEXT,
                    date_published TEXT,
                    law_category TEXT,
                    legal_references TEXT,
                    entities TEXT,
                    quality_score REAL,
                    language TEXT,
                    word_count INTEGER,
                    processed_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON documents(law_category)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_quality ON documents(quality_score)
            ''')
            
            conn.commit()
        
        logger.info("Database initialized")
    
    async def test_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all data sources"""
        
        connectivity_status = {}
        
        await self.connector.initialize()
        
        for source_name, source_config in self.config.data_sources.items():
            if not source_config.get('enabled', False):
                connectivity_status[source_name] = False
                continue
            
            try:
                url = source_config['url']
                content = await self.connector.fetch_url(url)
                connectivity_status[source_name] = content is not None and len(content) > 100
                
                if connectivity_status[source_name]:
                    logger.success(f"✓ {source_name}: Connected")
                else:
                    logger.warning(f"✗ {source_name}: Connection failed")
                    
            except Exception as e:
                logger.error(f"✗ {source_name}: {e}")
                connectivity_status[source_name] = False
            
            # Rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)
        
        await self.connector.cleanup()
        
        return connectivity_status
    
    async def start_collection(self) -> None:
        """Start the data collection process"""
        
        if self.collection_active:
            logger.warning("Collection already active")
            return
        
        try:
            logger.info("Starting Persian legal data collection...")
            self.collection_active = True
            
            await self.connector.initialize()
            
            # Collect from all enabled sources
            tasks = []
            for source_name, source_config in self.config.data_sources.items():
                if source_config.get('enabled', False):
                    task = asyncio.create_task(
                        self._collect_from_source(source_name, source_config)
                    )
                    tasks.append(task)
            
            # Wait for all collection tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                source_name = list(self.config.data_sources.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Collection from {source_name} failed: {result}")
                else:
                    logger.success(f"Collection from {source_name} completed: {result} documents")
            
            logger.success(f"Data collection completed! Total documents: {self.collected_documents}")
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
        finally:
            self.collection_active = False
            await self.connector.cleanup()
    
    async def _collect_from_source(self, source_name: str, source_config: Dict[str, Any]) -> int:
        """Collect documents from a specific source"""
        
        logger.info(f"Starting collection from {source_name}")
        
        documents_collected = 0
        max_documents = source_config.get('max_documents', 1000)
        
        try:
            if source_name == 'naab_corpus':
                documents_collected = await self._collect_naab_corpus(source_config, max_documents)
            elif source_name == 'iran_data_portal':
                documents_collected = await self._collect_iran_data_portal(source_config, max_documents)
            elif source_name == 'qavanin_portal':
                documents_collected = await self._collect_qavanin_portal(source_config, max_documents)
            elif source_name == 'majles_website':
                documents_collected = await self._collect_majles_website(source_config, max_documents)
            
            self.collection_stats['documents_by_source'][source_name] = documents_collected
            
        except Exception as e:
            logger.error(f"Error collecting from {source_name}: {e}")
        
        return documents_collected
    
    async def _collect_naab_corpus(self, config: Dict[str, Any], max_docs: int) -> int:
        """Collect from Naab Corpus (simulated - would need actual API)"""
        
        # In real implementation, this would connect to the actual Naab Corpus API
        # For demonstration, we simulate the collection process
        
        collected = 0
        base_url = config['url']
        
        # Simulate document collection
        for i in range(min(100, max_docs)):  # Simulate collecting 100 documents
            
            # Create synthetic document for demonstration
            document = LegalDocument(
                id=f"naab_{i:06d}",
                title=f"سند حقوقی شماره {i+1}",
                content=f"این یک نمونه متن حقوقی فارسی است که شامل مواد قانونی مختلف می‌باشد. ماده {i+1} - این ماده در مورد حقوق مدنی و تجاری صحبت می‌کند...",
                source="naab_corpus",
                url=f"{base_url}/document/{i+1}",
                document_type="legal_text"
            )
            
            # Process document
            processed_doc = await self._process_document(document)
            
            if processed_doc and processed_doc.quality_score >= self.config.quality_threshold:
                await self._save_document(processed_doc)
                collected += 1
                self.collected_documents += 1
            
            # Rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)
            
            if collected >= max_docs:
                break
        
        logger.info(f"Collected {collected} documents from Naab Corpus")
        return collected
    
    async def _collect_iran_data_portal(self, config: Dict[str, Any], max_docs: int) -> int:
        """Collect from Iran Data Portal"""
        
        collected = 0
        base_url = config['url']
        
        try:
            # Fetch main page
            main_content = await self.connector.fetch_url(base_url)
            if not main_content:
                return 0
            
            # Parse main page for document links
            soup = BeautifulSoup(main_content, 'html.parser')
            
            # Find document links (this would be customized based on actual site structure)
            document_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(keyword in href.lower() for keyword in ['law', 'legal', 'قانون', 'حقوق']):
                    full_url = urljoin(base_url, href)
                    document_links.append(full_url)
            
            # Collect documents
            for url in document_links[:max_docs]:
                try:
                    content = await self.connector.fetch_url(url)
                    if content:
                        # Extract document content
                        doc_soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in doc_soup(["script", "style"]):
                            script.decompose()
                        
                        title = doc_soup.title.string if doc_soup.title else "Untitled"
                        text_content = doc_soup.get_text()
                        
                        # Create document
                        document = LegalDocument(
                            id=hashlib.md5(url.encode()).hexdigest(),
                            title=title,
                            content=text_content,
                            source="iran_data_portal",
                            url=url,
                            document_type="legal_document"
                        )
                        
                        # Process document
                        processed_doc = await self._process_document(document)
                        
                        if processed_doc and processed_doc.quality_score >= self.config.quality_threshold:
                            await self._save_document(processed_doc)
                            collected += 1
                            self.collected_documents += 1
                
                except Exception as e:
                    logger.warning(f"Failed to collect document from {url}: {e}")
                    self.failed_urls += 1
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
                
                if collected >= max_docs:
                    break
        
        except Exception as e:
            logger.error(f"Error collecting from Iran Data Portal: {e}")
        
        logger.info(f"Collected {collected} documents from Iran Data Portal")
        return collected
    
    async def _collect_qavanin_portal(self, config: Dict[str, Any], max_docs: int) -> int:
        """Collect from Qavanin Portal"""
        
        # Similar implementation to Iran Data Portal
        # This would be customized based on the actual site structure
        
        collected = 0
        # Simulation for demonstration
        for i in range(min(50, max_docs)):
            document = LegalDocument(
                id=f"qavanin_{i:06d}",
                title=f"قانون شماره {i+1}",
                content=f"متن کامل قانون شماره {i+1} که شامل مواد مختلف حقوقی است...",
                source="qavanin_portal",
                url=f"{config['url']}/law/{i+1}",
                document_type="law"
            )
            
            processed_doc = await self._process_document(document)
            if processed_doc and processed_doc.quality_score >= self.config.quality_threshold:
                await self._save_document(processed_doc)
                collected += 1
                self.collected_documents += 1
            
            await asyncio.sleep(self.config.rate_limit_delay)
        
        logger.info(f"Collected {collected} documents from Qavanin Portal")
        return collected
    
    async def _collect_majles_website(self, config: Dict[str, Any], max_docs: int) -> int:
        """Collect from Majles Website"""
        
        # Similar implementation pattern
        collected = 0
        
        # Simulation for demonstration
        for i in range(min(30, max_docs)):
            document = LegalDocument(
                id=f"majles_{i:06d}",
                title=f"مصوبه مجلس شماره {i+1}",
                content=f"متن کامل مصوبه مجلس که در جلسه شماره {i+1} تصویب شده است...",
                source="majles_website",
                url=f"{config['url']}/bill/{i+1}",
                document_type="parliamentary_bill"
            )
            
            processed_doc = await self._process_document(document)
            if processed_doc and processed_doc.quality_score >= self.config.quality_threshold:
                await self._save_document(processed_doc)
                collected += 1
                self.collected_documents += 1
            
            await asyncio.sleep(self.config.rate_limit_delay)
        
        logger.info(f"Collected {collected} documents from Majles Website")
        return collected
    
    async def _process_document(self, document: LegalDocument) -> Optional[LegalDocument]:
        """Process and enhance a legal document"""
        
        try:
            # Normalize text
            normalized_content = self.text_processor.normalize_text(document.content)
            
            # Skip if content is too short or too long
            if (len(normalized_content) < self.config.min_document_length or 
                len(normalized_content) > self.config.max_document_length):
                return None
            
            # Extract entities
            entities = self.text_processor.extract_legal_entities(normalized_content)
            
            # Assess quality
            quality_score = self.text_processor.assess_quality(normalized_content)
            
            # Categorize document
            category = self.text_processor.categorize_document(normalized_content)
            
            # Update document
            document.content = normalized_content
            document.entities = entities
            document.quality_score = quality_score
            document.law_category = category
            document.word_count = len(normalized_content.split())
            
            # Extract legal references
            references = []
            for match in self.text_processor.law_patterns['law_reference'].finditer(normalized_content):
                references.append(f"{match.group(1)} ({match.group(2)})")
            document.legal_references = references
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            return None
    
    async def _save_document(self, document: LegalDocument) -> None:
        """Save document to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for duplicates
                cursor.execute('SELECT id FROM documents WHERE id = ?', (document.id,))
                if cursor.fetchone():
                    self.duplicate_count += 1
                    return
                
                # Insert document
                cursor.execute('''
                    INSERT INTO documents (
                        id, title, content, source, url, document_type,
                        date_published, law_category, legal_references,
                        entities, quality_score, language, word_count, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    document.id,
                    document.title,
                    document.content,
                    document.source,
                    document.url,
                    document.document_type,
                    document.date_published,
                    document.law_category,
                    json.dumps(document.legal_references),
                    json.dumps(document.entities),
                    document.quality_score,
                    document.language,
                    document.word_count,
                    document.processed_at
                ))
                
                conn.commit()
            
            # Update statistics
            self.collection_stats['quality_distribution'].append(document.quality_score)
            
            category = document.law_category or 'unknown'
            if category not in self.collection_stats['documents_by_category']:
                self.collection_stats['documents_by_category'][category] = 0
            self.collection_stats['documents_by_category'][category] += 1
            
        except Exception as e:
            logger.error(f"Error saving document {document.id}: {e}")
    
    async def stop(self) -> None:
        """Stop data collection"""
        self.collection_active = False
        await self.connector.cleanup()
        logger.info("Data collection stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        
        return {
            'total_documents': self.collected_documents,
            'failed_urls': self.failed_urls,
            'duplicates': self.duplicate_count,
            'sources': self.collection_stats['documents_by_source'],
            'categories': self.collection_stats['documents_by_category'],
            'average_quality': np.mean(self.collection_stats['quality_distribution']) if self.collection_stats['quality_distribution'] else 0.0,
            'collection_active': self.collection_active
        }
    
    def get_documents(self, limit: int = 100, min_quality: float = 0.5) -> List[Dict[str, Any]]:
        """Retrieve documents from database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM documents 
                    WHERE quality_score >= ? 
                    ORDER BY quality_score DESC 
                    LIMIT ?
                ''', (min_quality, limit))
                
                columns = [description[0] for description in cursor.description]
                documents = []
                
                for row in cursor.fetchall():
                    doc_dict = dict(zip(columns, row))
                    # Parse JSON fields
                    if doc_dict['legal_references']:
                        doc_dict['legal_references'] = json.loads(doc_dict['legal_references'])
                    if doc_dict['entities']:
                        doc_dict['entities'] = json.loads(doc_dict['entities'])
                    documents.append(doc_dict)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []