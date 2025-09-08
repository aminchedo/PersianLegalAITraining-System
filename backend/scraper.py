import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
import random
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class LegalScraper:
    def __init__(self):
        self.session = None
        self.proxies = []
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
    async def __aenter__(self):
        await self.init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def init_session(self):
        """Initialize aiohttp session with proper configuration"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'Accept-Language': 'fa-IR,fa;q=0.9,en;q=0.8'}
        )
        
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
    
    def get_random_headers(self) -> Dict[str, str]:
        """Get randomized headers to avoid detection"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fa-IR,fa;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def scrape_site(self, domain: str) -> List[Dict]:
        """Main scraping method for Iranian legal sites"""
        if not self.session:
            await self.init_session()
            
        try:
            if domain == 'divan-edalat.ir':
                return await self.scrape_divan_edalat()
            elif domain == 'majlis.ir':
                return await self.scrape_majlis()
            elif domain == 'president.ir':
                return await self.scrape_president()
            elif domain == 'judiciary.ir':
                return await self.scrape_judiciary()
            elif domain == 'rrk.ir':
                return await self.scrape_rrk()
            elif domain == 'shora-gc.ir':
                return await self.scrape_shora_gc()
            else:
                logger.warning(f"Unknown domain: {domain}")
                return []
                
        except Exception as e:
            logger.error(f"Error scraping {domain}: {str(e)}")
            return []
    
    async def scrape_divan_edalat(self) -> List[Dict]:
        """Scrape Supreme Administrative Court (دیوان عدالت اداری)"""
        documents = []
        base_url = 'https://divan-edalat.ir'
        
        try:
            # Get main page
            async with self.session.get(
                f'{base_url}/fa/news',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch divan-edalat main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find news/decision links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['news', 'decision', 'ruling', 'verdict']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'divan-edalat.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping divan-edalat: {str(e)}")
            
        return documents
    
    async def scrape_majlis(self) -> List[Dict]:
        """Scrape Islamic Parliament (مجلس شورای اسلامی)"""
        documents = []
        base_url = 'https://majlis.ir'
        
        try:
            # Get legislation page
            async with self.session.get(
                f'{base_url}/fa/law',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch majlis main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find law/bill links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['law', 'bill', 'legislation']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'majlis.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping majlis: {str(e)}")
            
        return documents
    
    async def scrape_president(self) -> List[Dict]:
        """Scrape Presidential Office (ریاست جمهوری)"""
        documents = []
        base_url = 'https://president.ir'
        
        try:
            # Get decrees/orders page
            async with self.session.get(
                f'{base_url}/fa/decrees',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch president main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find decree/order links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['decree', 'order', 'directive']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'president.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping president: {str(e)}")
            
        return documents
    
    async def scrape_judiciary(self) -> List[Dict]:
        """Scrape Judiciary (قوه قضائیه)"""
        documents = []
        base_url = 'https://judiciary.ir'
        
        try:
            # Get rulings page
            async with self.session.get(
                f'{base_url}/fa/rulings',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch judiciary main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find ruling links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['ruling', 'judgment', 'decision']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'judiciary.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping judiciary: {str(e)}")
            
        return documents
    
    async def scrape_rrk(self) -> List[Dict]:
        """Scrape Council of Guardians (شورای نگهبان)"""
        documents = []
        base_url = 'https://rrk.ir'
        
        try:
            # Get opinions page
            async with self.session.get(
                f'{base_url}/fa/opinions',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch rrk main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find opinion links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['opinion', 'interpretation', 'review']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'rrk.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping rrk: {str(e)}")
            
        return documents
    
    async def scrape_shora_gc(self) -> List[Dict]:
        """Scrape Guardian Council (شورای نگهبان قانون اساسی)"""
        documents = []
        base_url = 'https://shora-gc.ir'
        
        try:
            # Get constitutional reviews page
            async with self.session.get(
                f'{base_url}/fa/reviews',
                headers=self.get_random_headers()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch shora-gc main page: {response.status}")
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find review links
                links = soup.find_all('a', href=True)
                document_links = []
                
                for link in links:
                    href = link.get('href', '')
                    if any(keyword in href.lower() for keyword in ['review', 'constitutional', 'examination']):
                        full_url = urljoin(base_url, href)
                        document_links.append(full_url)
                
                # Scrape individual documents
                for url in document_links[:10]:  # Limit for demo
                    try:
                        doc = await self.scrape_document_page(url, 'shora-gc.ir')
                        if doc:
                            documents.append(doc)
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error scraping document {url}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error scraping shora-gc: {str(e)}")
            
        return documents
    
    async def scrape_document_page(self, url: str, source: str) -> Optional[Dict]:
        """Scrape individual document page"""
        try:
            async with self.session.get(url, headers=self.get_random_headers()) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch document {url}: {response.status}")
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = self.extract_title(soup)
                if not title:
                    logger.warning(f"No title found for document {url}")
                    return None
                
                # Extract content
                content = self.extract_content(soup)
                if not content or len(content.strip()) < 100:
                    logger.warning(f"Insufficient content for document {url}")
                    return None
                
                # Clean and validate content
                content = self.clean_persian_text(content)
                title = self.clean_persian_text(title)
                
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'source': source,
                    'scraped_at': datetime.now().isoformat(),
                    'metadata': {
                        'content_length': len(content),
                        'scraper_version': '1.0.0'
                    }
                }
                
        except Exception as e:
            logger.error(f"Error scraping document page {url}: {str(e)}")
            return None
    
    def extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract document title from HTML"""
        # Try different title selectors
        title_selectors = [
            'h1',
            '.title',
            '.document-title',
            '.news-title',
            '.post-title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 10:
                    return title
        
        return None
    
    def extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract document content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try different content selectors
        content_selectors = [
            '.content',
            '.document-content',
            '.news-content',
            '.post-content',
            '.article-content',
            'main',
            '.main-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator=' ', strip=True)
                if content and len(content) > 100:
                    return content
        
        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            if len(content) > 100:
                return content
        
        # Last resort: get body text
        body = soup.find('body')
        if body:
            content = body.get_text(separator=' ', strip=True)
            if len(content) > 100:
                return content
        
        return None
    
    def clean_persian_text(self, text: str) -> str:
        """Clean and normalize Persian text"""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted characters but keep Persian, Arabic, and common punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\w\.\,\:\;\!\?\-\(\)\[\]\"\']+', '', text)
        
        # Normalize Persian/Arabic characters
        persian_chars = {
            'ي': 'ی',
            'ك': 'ک',
            'ء': 'ٔ',
        }
        
        for old_char, new_char in persian_chars.items():
            text = text.replace(old_char, new_char)
        
        return text.strip()