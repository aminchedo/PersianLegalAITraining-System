"""
Dataset Quality Validator for Persian Legal AI
اعتبارسنج کیفیت مجموعه داده‌ها برای هوش مصنوعی حقوقی فارسی

Validates dataset quality with REAL checks - no fake validation.
Ensures only authentic, high-quality datasets are used for training.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

class DatasetQualityValidator:
    """Validates dataset quality with REAL checks - no fake validation"""
    
    def __init__(self):
        self.validation_results = {}
        self.persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        self.legal_keywords = [
            'قانون', 'حقوق', 'دادگاه', 'قاضی', 'وکیل', 'محکمه', 'حکم', 'رأی',
            'ماده', 'بند', 'تبصره', 'مقررات', 'آیین‌نامه', 'دستورالعمل',
            'شکایت', 'دعوا', 'محاکمه', 'محکومیت', 'برائت', 'جریمه',
            'قانون‌گذاری', 'مجلس', 'شورای', 'ملی', 'اسلامی', 'جمهوری',
            'دستور', 'فرمان', 'بخشنامه', 'راهنمای', 'راهکار'
        ]
        
        logger.info("DatasetQualityValidator initialized")
    
    def validate_datasets(self, datasets: Dict[str, Any]) -> Dict[str, bool]:
        """
        Real validation using actual quality metrics
        
        Args:
            datasets: Dictionary of datasets to validate
            
        Returns:
            Dictionary mapping dataset keys to validation results
        """
        logger.info(f"Starting validation of {len(datasets)} datasets")
        validation_results = {}
        
        for key, dataset in datasets.items():
            try:
                logger.info(f"Validating dataset: {key}")
                is_valid = self._perform_real_validation(dataset, key)
                validation_results[key] = is_valid
                
                status = "✅ PASS" if is_valid else "❌ FAIL"
                logger.info(f"{status} - Dataset {key}")
                
            except Exception as e:
                logger.error(f"Validation failed for {key}: {str(e)}")
                validation_results[key] = False
        
        self.validation_results = validation_results
        return validation_results
    
    def _perform_real_validation(self, dataset: Any, dataset_key: str) -> bool:
        """Actual validation logic - no mock checks"""
        try:
            # Perform comprehensive validation checks
            validation_checks = {
                'legal_term_density': self._check_legal_term_density(dataset),
                'text_quality_score': self._check_text_quality_score(dataset),
                'source_authenticity': self._check_source_authenticity(dataset),
                'content_originality': self._check_content_originality(dataset),
                'persian_language_quality': self._check_persian_language_quality(dataset),
                'format_consistency': self._check_format_consistency(dataset)
            }
            
            # Log detailed results
            logger.info(f"Validation results for {dataset_key}:")
            for check_name, result in validation_checks.items():
                logger.info(f"  {check_name}: {result}")
            
            # Determine overall validity
            # At least 4 out of 6 checks must pass
            passed_checks = sum(1 for result in validation_checks.values() if result)
            total_checks = len(validation_checks)
            
            is_valid = passed_checks >= 4
            logger.info(f"Overall validation: {passed_checks}/{total_checks} checks passed")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error during validation of {dataset_key}: {e}")
            return False
    
    def _check_legal_term_density(self, dataset: Any) -> bool:
        """Check density of legal terms in the dataset"""
        try:
            legal_term_counts = []
            sample_count = 0
            
            for sample in dataset.take(100):  # Check first 100 samples
                sample_count += 1
                text_content = self._extract_text_from_sample(sample)
                
                if text_content:
                    # Count legal keywords
                    legal_count = sum(1 for keyword in self.legal_keywords if keyword in text_content)
                    text_length = len(text_content)
                    
                    if text_length > 0:
                        legal_density = legal_count / text_length * 1000  # Per 1000 characters
                        legal_term_counts.append(legal_density)
            
            if not legal_term_counts:
                return False
            
            # Calculate average legal term density
            avg_density = statistics.mean(legal_term_counts)
            logger.info(f"Average legal term density: {avg_density:.2f} per 1000 chars")
            
            # Require at least 0.3 legal terms per 1000 characters
            return avg_density >= 0.3
            
        except Exception as e:
            logger.error(f"Error checking legal term density: {e}")
            return False
    
    def _check_text_quality_score(self, dataset: Any) -> bool:
        """Check overall text quality using multiple metrics"""
        try:
            quality_scores = []
            sample_count = 0
            
            for sample in dataset.take(50):  # Check first 50 samples
                sample_count += 1
                text_content = self._extract_text_from_sample(sample)
                
                if text_content and len(text_content.strip()) > 20:
                    quality_score = self._calculate_text_quality_score(text_content)
                    quality_scores.append(quality_score)
            
            if not quality_scores:
                return False
            
            avg_quality = statistics.mean(quality_scores)
            logger.info(f"Average text quality score: {avg_quality:.2f}")
            
            # Require average quality score of at least 0.6
            return avg_quality >= 0.6
            
        except Exception as e:
            logger.error(f"Error checking text quality: {e}")
            return False
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate quality score for a single text"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.0
            
            score = 0.0
            
            # Length score (prefer texts between 50-2000 characters)
            length = len(text)
            if 50 <= length <= 2000:
                score += 0.3
            elif 20 <= length < 50 or 2000 < length <= 5000:
                score += 0.2
            else:
                score += 0.1
            
            # Persian character ratio
            persian_count = sum(1 for char in text if char in self.persian_chars)
            persian_ratio = persian_count / len(text) if len(text) > 0 else 0
            if persian_ratio >= 0.5:
                score += 0.3
            elif persian_ratio >= 0.3:
                score += 0.2
            else:
                score += 0.1
            
            # Sentence structure (basic check)
            sentences = re.split(r'[.!?]', text)
            if len(sentences) >= 2:
                score += 0.2
            
            # Word diversity (basic check)
            words = text.split()
            unique_words = set(words)
            if len(unique_words) / len(words) >= 0.5:
                score += 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating text quality score: {e}")
            return 0.0
    
    def _check_source_authenticity(self, dataset: Any) -> bool:
        """Check if dataset appears to be from authentic sources"""
        try:
            # This is a basic check - in a real implementation, you might
            # check metadata, source URLs, or other authenticity indicators
            sample_count = 0
            authentic_samples = 0
            
            for sample in dataset.take(30):
                sample_count += 1
                text_content = self._extract_text_from_sample(sample)
                
                if text_content:
                    # Check for signs of authentic legal content
                    authenticity_indicators = [
                        # Legal document patterns
                        'ماده' in text_content and 'قانون' in text_content,
                        'دادگاه' in text_content and 'قاضی' in text_content,
                        'رأی' in text_content and 'حکم' in text_content,
                        # Official document patterns
                        'دستور' in text_content or 'فرمان' in text_content,
                        'آیین‌نامه' in text_content or 'دستورالعمل' in text_content
                    ]
                    
                    if any(authenticity_indicators):
                        authentic_samples += 1
            
            authenticity_ratio = authentic_samples / sample_count if sample_count > 0 else 0
            logger.info(f"Source authenticity ratio: {authenticity_ratio:.2%}")
            
            # Require at least 40% of samples to show authenticity indicators
            return authenticity_ratio >= 0.4
            
        except Exception as e:
            logger.error(f"Error checking source authenticity: {e}")
            return False
    
    def _check_content_originality(self, dataset: Any) -> bool:
        """Check for content originality (basic duplicate detection)"""
        try:
            text_hashes = set()
            duplicate_count = 0
            sample_count = 0
            
            for sample in dataset.take(100):
                sample_count += 1
                text_content = self._extract_text_from_sample(sample)
                
                if text_content:
                    # Simple hash-based duplicate detection
                    text_hash = hash(text_content.strip().lower())
                    
                    if text_hash in text_hashes:
                        duplicate_count += 1
                    else:
                        text_hashes.add(text_hash)
            
            originality_ratio = 1.0 - (duplicate_count / sample_count) if sample_count > 0 else 0
            logger.info(f"Content originality ratio: {originality_ratio:.2%}")
            
            # Require at least 90% original content
            return originality_ratio >= 0.9
            
        except Exception as e:
            logger.error(f"Error checking content originality: {e}")
            return False
    
    def _check_persian_language_quality(self, dataset: Any) -> bool:
        """Check Persian language quality and correctness"""
        try:
            quality_scores = []
            sample_count = 0
            
            for sample in dataset.take(50):
                sample_count += 1
                text_content = self._extract_text_from_sample(sample)
                
                if text_content:
                    persian_quality = self._calculate_persian_quality(text_content)
                    quality_scores.append(persian_quality)
            
            if not quality_scores:
                return False
            
            avg_persian_quality = statistics.mean(quality_scores)
            logger.info(f"Average Persian language quality: {avg_persian_quality:.2f}")
            
            # Require average Persian quality of at least 0.7
            return avg_persian_quality >= 0.7
            
        except Exception as e:
            logger.error(f"Error checking Persian language quality: {e}")
            return False
    
    def _calculate_persian_quality(self, text: str) -> float:
        """Calculate Persian language quality score"""
        try:
            if not text:
                return 0.0
            
            score = 0.0
            
            # Persian character ratio
            persian_count = sum(1 for char in text if char in self.persian_chars)
            persian_ratio = persian_count / len(text) if len(text) > 0 else 0
            score += persian_ratio * 0.4
            
            # Proper Persian word patterns (basic check)
            words = text.split()
            persian_words = [word for word in words if any(char in self.persian_chars for char in word)]
            if len(persian_words) / len(words) >= 0.6:
                score += 0.3
            
            # Sentence structure (Persian sentences typically end with proper punctuation)
            if text.strip().endswith(('.', '!', '؟', '?')):
                score += 0.2
            
            # Character diversity (avoid repetitive patterns)
            unique_chars = len(set(text))
            if unique_chars / len(text) >= 0.3:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating Persian quality: {e}")
            return 0.0
    
    def _check_format_consistency(self, dataset: Any) -> bool:
        """Check if dataset has consistent format across samples"""
        try:
            sample_formats = []
            sample_count = 0
            
            for sample in dataset.take(30):
                sample_count += 1
                
                if isinstance(sample, dict):
                    # Check if it has consistent structure
                    format_info = {
                        'type': 'dict',
                        'keys': list(sample.keys()),
                        'has_text': any(isinstance(v, str) and len(v.strip()) > 10 for v in sample.values())
                    }
                elif isinstance(sample, str):
                    format_info = {
                        'type': 'string',
                        'length': len(sample),
                        'has_text': len(sample.strip()) > 10
                    }
                else:
                    format_info = {
                        'type': 'other',
                        'has_text': False
                    }
                
                sample_formats.append(format_info)
            
            # Check consistency
            if not sample_formats:
                return False
            
            # Count samples with text content
            text_samples = sum(1 for fmt in sample_formats if fmt.get('has_text', False))
            text_ratio = text_samples / len(sample_formats)
            
            # Check type consistency
            types = [fmt['type'] for fmt in sample_formats]
            most_common_type = max(set(types), key=types.count)
            type_consistency = types.count(most_common_type) / len(types)
            
            logger.info(f"Format consistency - Text ratio: {text_ratio:.2%}, Type consistency: {type_consistency:.2%}")
            
            # Require at least 80% text samples and 70% type consistency
            return text_ratio >= 0.8 and type_consistency >= 0.7
            
        except Exception as e:
            logger.error(f"Error checking format consistency: {e}")
            return False
    
    def _extract_text_from_sample(self, sample: Any) -> str:
        """Extract text content from a dataset sample"""
        if isinstance(sample, dict):
            # Find the text field
            for key, value in sample.items():
                if isinstance(value, str) and len(value.strip()) > 10:
                    return value
            # If no single text field, concatenate all string values
            text_parts = [str(value) for value in sample.values() if isinstance(value, str)]
            return " ".join(text_parts)
        elif isinstance(sample, str):
            return sample
        else:
            return str(sample)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        return {
            'validation_results': self.validation_results,
            'total_datasets': len(self.validation_results),
            'passed_datasets': sum(1 for result in self.validation_results.values() if result),
            'failed_datasets': sum(1 for result in self.validation_results.values() if not result),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_dataset_quality_metrics(self, dataset: Any, dataset_key: str) -> Dict[str, Any]:
        """Get detailed quality metrics for a specific dataset"""
        try:
            metrics = {
                'dataset_key': dataset_key,
                'legal_term_density': self._check_legal_term_density(dataset),
                'text_quality_score': self._check_text_quality_score(dataset),
                'source_authenticity': self._check_source_authenticity(dataset),
                'content_originality': self._check_content_originality(dataset),
                'persian_language_quality': self._check_persian_language_quality(dataset),
                'format_consistency': self._check_format_consistency(dataset),
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate overall quality score
            quality_scores = [v for v in metrics.values() if isinstance(v, bool)]
            metrics['overall_quality_score'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting quality metrics for {dataset_key}: {e}")
            return {'error': str(e), 'dataset_key': dataset_key}