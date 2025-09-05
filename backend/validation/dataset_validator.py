"""
Dataset Quality Validation System
سیستم اعتبارسنجی کیفیت مجموعه داده‌ها
"""

import re
import unicodedata
from typing import Dict, List, Optional, Any, Tuple
import logging
from datasets import Dataset
import numpy as np
from collections import Counter
import json

logger = logging.getLogger(__name__)

class DatasetQualityValidator:
    """Comprehensive validation system for Persian legal datasets"""
    
    def __init__(self):
        # Persian legal keywords for relevance checking
        self.legal_keywords = [
            'قانون', 'حقوق', 'دادگاه', 'قاضی', 'مواد', 'احکام', 
            'قضایی', 'شورای', 'مجلس', 'تصویب', 'دادرسی', 'حکم',
            'رای', 'قضاوت', 'عدالت', 'حق', 'تعهد', 'قرارداد',
            'ماده', 'بند', 'تبصره', 'مقررات', 'آیین‌نامه', 'دستورالعمل',
            'بخشنامه', 'تصمیم', 'رأی', 'حکمیت', 'داوری', 'صلح',
            'مصالحه', 'جریمه', 'مجازات', 'تعزیر', 'حد', 'قصاص',
            'دیات', 'ارش', 'ضمان', 'مسئولیت', 'خسارت', 'غرامت',
            'وصیت', 'ارث', 'نکاح', 'طلاق', 'مهریه', 'نفقه',
            'حضانت', 'ولایت', 'قیمومت', 'اموال', 'مالکیت', 'تصرف',
            'اجاره', 'بیع', 'شرط', 'شرایط', 'مشروط', 'مطلق',
            'مشروطیت', 'مطلقی', 'شرطی', 'مشروطی', 'مطلقی'
        ]
        
        # Persian stop words
        self.persian_stop_words = [
            'در', 'از', 'به', 'با', 'که', 'این', 'آن', 'را', 'است', 'بود',
            'می', 'خواهد', 'کرد', 'کرده', 'شده', 'می‌شود', 'می‌کند',
            'باشد', 'باشند', 'بوده', 'بودند', 'خواهد', 'خواهند',
            'کرده', 'کردند', 'شده', 'شده‌اند', 'می‌شود', 'می‌شوند'
        ]
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_length': 50,
            'max_length': 10000,
            'min_persian_ratio': 0.3,
            'min_legal_keywords_ratio': 0.1,
            'max_repetition_ratio': 0.3,
            'min_uniqueness_ratio': 0.7
        }
    
    def validate_datasets(self, datasets: Dict[str, Dataset], 
                         sample_size: int = 1000) -> Dict[str, Dict]:
        """Comprehensive validation of all datasets"""
        logger.info(f"🔍 Starting comprehensive dataset validation (sample size: {sample_size})...")
        
        validation_results = {}
        
        for key, dataset in datasets.items():
            logger.info(f"Validating dataset: {key}")
            try:
                result = self._validate_single_dataset(dataset, key, sample_size)
                validation_results[key] = result
                
                if result['overall_quality_score'] >= 70:
                    logger.info(f"✅ Dataset {key} passed quality validation (Score: {result['overall_quality_score']}/100)")
                else:
                    logger.warning(f"⚠️ Dataset {key} failed quality validation (Score: {result['overall_quality_score']}/100)")
                    
            except Exception as e:
                logger.error(f"❌ Error validating {key}: {e}")
                validation_results[key] = {
                    'overall_quality_score': 0,
                    'error': str(e),
                    'passed': False
                }
        
        # Generate summary
        summary = self._generate_validation_summary(validation_results)
        validation_results['_summary'] = summary
        
        logger.info(f"📊 Validation completed. {summary['passed_count']}/{summary['total_count']} datasets passed")
        
        return validation_results
    
    def _validate_single_dataset(self, dataset: Dataset, dataset_key: str, 
                                sample_size: int) -> Dict:
        """Validate a single dataset's quality"""
        logger.info(f"Validating {dataset_key} with {len(dataset)} samples...")
        
        # Sample data for validation
        actual_sample_size = min(sample_size, len(dataset))
        sample_indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
        sample_data = dataset.select(sample_indices)
        
        validation_results = {
            'dataset_key': dataset_key,
            'total_samples': len(dataset),
            'validated_samples': actual_sample_size,
            'checks': {}
        }
        
        # Perform all quality checks
        checks = [
            ('minimum_size', self._check_minimum_size, (dataset,)),
            ('text_quality', self._check_text_quality, (sample_data,)),
            ('legal_relevance', self._check_legal_relevance, (sample_data, dataset_key)),
            ('format_consistency', self._check_format_consistency, (sample_data,)),
            ('persian_language', self._check_persian_language, (sample_data,)),
            ('content_uniqueness', self._check_content_uniqueness, (sample_data,)),
            ('text_diversity', self._check_text_diversity, (sample_data,)),
            ('encoding_quality', self._check_encoding_quality, (sample_data,))
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func, args in checks:
            try:
                result = check_func(*args)
                validation_results['checks'][check_name] = result
                
                if result.get('passed', False):
                    passed_checks += 1
                    logger.debug(f"✅ {check_name}: PASSED")
                else:
                    logger.debug(f"❌ {check_name}: FAILED - {result.get('message', '')}")
                    
            except Exception as e:
                logger.error(f"❌ Error in {check_name}: {e}")
                validation_results['checks'][check_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Calculate overall quality score
        quality_score = (passed_checks / total_checks) * 100
        validation_results['overall_quality_score'] = round(quality_score, 2)
        validation_results['passed'] = quality_score >= 70
        
        return validation_results
    
    def _check_minimum_size(self, dataset: Dataset) -> Dict:
        """Check if dataset meets minimum size requirements"""
        min_samples = 1000
        
        return {
            'passed': len(dataset) >= min_samples,
            'actual_samples': len(dataset),
            'required_samples': min_samples,
            'message': f"Dataset has {len(dataset)} samples (required: {min_samples})"
        }
    
    def _check_text_quality(self, sample_data: Dataset) -> Dict:
        """Check text quality metrics"""
        valid_samples = 0
        total_length = 0
        length_violations = 0
        
        for sample in sample_data:
            text = sample.get('text', '')
            
            if not text or not isinstance(text, str):
                continue
            
            text = text.strip()
            text_length = len(text)
            
            # Check length requirements
            if (text_length >= self.quality_thresholds['min_length'] and 
                text_length <= self.quality_thresholds['max_length']):
                valid_samples += 1
                total_length += text_length
            else:
                length_violations += 1
        
        avg_length = total_length / valid_samples if valid_samples > 0 else 0
        quality_ratio = valid_samples / len(sample_data) if len(sample_data) > 0 else 0
        
        return {
            'passed': quality_ratio >= 0.8,  # 80% of samples should be valid
            'valid_samples': valid_samples,
            'total_samples': len(sample_data),
            'quality_ratio': quality_ratio,
            'avg_length': avg_length,
            'length_violations': length_violations,
            'message': f"{valid_samples}/{len(sample_data)} samples passed quality checks"
        }
    
    def _check_legal_relevance(self, sample_data: Dataset, dataset_key: str) -> Dict:
        """Ensure dataset contains relevant legal content"""
        legal_matches = 0
        total_samples = len(sample_data)
        
        for sample in sample_data:
            text = sample.get('text', '')
            if not text:
                continue
            
            # Count legal keywords
            keyword_count = sum(1 for keyword in self.legal_keywords if keyword in text)
            
            if keyword_count >= 2:  # At least 2 legal keywords
                legal_matches += 1
        
        legal_ratio = legal_matches / total_samples if total_samples > 0 else 0
        
        return {
            'passed': legal_ratio >= self.quality_thresholds['min_legal_keywords_ratio'],
            'legal_matches': legal_matches,
            'total_samples': total_samples,
            'legal_ratio': legal_ratio,
            'message': f"{legal_matches}/{total_samples} samples contain legal content"
        }
    
    def _check_format_consistency(self, sample_data: Dataset) -> Dict:
        """Check if dataset has consistent format"""
        format_issues = 0
        required_fields = ['text']
        
        for sample in sample_data:
            # Check required fields
            if not all(field in sample for field in required_fields):
                format_issues += 1
                continue
            
            # Check field types
            if not isinstance(sample.get('text'), str):
                format_issues += 1
        
        consistency_ratio = 1 - (format_issues / len(sample_data)) if len(sample_data) > 0 else 0
        
        return {
            'passed': consistency_ratio >= 0.95,  # 95% consistency required
            'format_issues': format_issues,
            'consistency_ratio': consistency_ratio,
            'message': f"{format_issues} format issues found in {len(sample_data)} samples"
        }
    
    def _check_persian_language(self, sample_data: Dataset) -> Dict:
        """Check if dataset contains Persian language content"""
        persian_samples = 0
        total_chars = 0
        persian_chars = 0
        
        for sample in sample_data:
            text = sample.get('text', '')
            if not text:
                continue
            
            # Count Persian characters
            text_chars = len(text)
            persian_char_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
            
            total_chars += text_chars
            persian_chars += persian_char_count
            
            # Check if sample is primarily Persian
            if text_chars > 0 and (persian_char_count / text_chars) >= self.quality_thresholds['min_persian_ratio']:
                persian_samples += 1
        
        persian_ratio = persian_chars / total_chars if total_chars > 0 else 0
        sample_persian_ratio = persian_samples / len(sample_data) if len(sample_data) > 0 else 0
        
        return {
            'passed': persian_ratio >= self.quality_thresholds['min_persian_ratio'],
            'persian_samples': persian_samples,
            'total_samples': len(sample_data),
            'persian_ratio': persian_ratio,
            'sample_persian_ratio': sample_persian_ratio,
            'message': f"{persian_samples}/{len(sample_data)} samples are primarily Persian"
        }
    
    def _check_content_uniqueness(self, sample_data: Dataset) -> Dict:
        """Check for content uniqueness and avoid duplicates"""
        texts = []
        duplicates = 0
        
        for sample in sample_data:
            text = sample.get('text', '').strip()
            if text:
                if text in texts:
                    duplicates += 1
                else:
                    texts.append(text)
        
        uniqueness_ratio = 1 - (duplicates / len(sample_data)) if len(sample_data) > 0 else 0
        
        return {
            'passed': uniqueness_ratio >= self.quality_thresholds['min_uniqueness_ratio'],
            'duplicates': duplicates,
            'unique_texts': len(texts),
            'uniqueness_ratio': uniqueness_ratio,
            'message': f"Found {duplicates} duplicate texts out of {len(sample_data)} samples"
        }
    
    def _check_text_diversity(self, sample_data: Dataset) -> Dict:
        """Check text diversity and vocabulary richness"""
        all_texts = []
        word_counts = Counter()
        
        for sample in sample_data:
            text = sample.get('text', '')
            if text:
                all_texts.append(text)
                # Simple word tokenization (split by whitespace)
                words = text.split()
                word_counts.update(words)
        
        # Calculate diversity metrics
        total_words = sum(word_counts.values())
        unique_words = len(word_counts)
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Check for repetitive content
        avg_text_length = np.mean([len(text) for text in all_texts]) if all_texts else 0
        
        return {
            'passed': vocabulary_richness >= 0.1,  # At least 10% unique words
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_richness': vocabulary_richness,
            'avg_text_length': avg_text_length,
            'message': f"Vocabulary richness: {vocabulary_richness:.3f} ({unique_words}/{total_words} unique words)"
        }
    
    def _check_encoding_quality(self, sample_data: Dataset) -> Dict:
        """Check text encoding and character quality"""
        encoding_issues = 0
        total_chars = 0
        
        for sample in sample_data:
            text = sample.get('text', '')
            if not text:
                continue
            
            try:
                # Check for encoding issues
                text.encode('utf-8')
                
                # Check for problematic characters
                for char in text:
                    total_chars += 1
                    if ord(char) > 0x10FFFF:  # Invalid Unicode
                        encoding_issues += 1
                    elif unicodedata.category(char) == 'Cc':  # Control characters
                        encoding_issues += 1
                        
            except UnicodeEncodeError:
                encoding_issues += 1
        
        quality_ratio = 1 - (encoding_issues / total_chars) if total_chars > 0 else 1
        
        return {
            'passed': quality_ratio >= 0.99,  # 99% encoding quality required
            'encoding_issues': encoding_issues,
            'total_chars': total_chars,
            'quality_ratio': quality_ratio,
            'message': f"Found {encoding_issues} encoding issues out of {total_chars} characters"
        }
    
    def _generate_validation_summary(self, validation_results: Dict) -> Dict:
        """Generate summary of validation results"""
        total_count = len([k for k in validation_results.keys() if not k.startswith('_')])
        passed_count = sum(1 for result in validation_results.values() 
                          if isinstance(result, dict) and result.get('passed', False))
        
        avg_quality_score = np.mean([
            result.get('overall_quality_score', 0) 
            for result in validation_results.values() 
            if isinstance(result, dict) and 'overall_quality_score' in result
        ])
        
        return {
            'total_count': total_count,
            'passed_count': passed_count,
            'failed_count': total_count - passed_count,
            'pass_rate': (passed_count / total_count * 100) if total_count > 0 else 0,
            'average_quality_score': round(avg_quality_score, 2)
        }
    
    def export_validation_report(self, validation_results: Dict, 
                               output_path: str = "./validation_report.json") -> str:
        """Export detailed validation report"""
        report = {
            'timestamp': self._get_timestamp(),
            'validation_summary': validation_results.get('_summary', {}),
            'dataset_results': {k: v for k, v in validation_results.items() if not k.startswith('_')},
            'quality_thresholds': self.quality_thresholds,
            'legal_keywords_count': len(self.legal_keywords)
        }
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Validation report exported to {output_path}")
        return output_path
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_recommendations(self, validation_results: Dict) -> List[str]:
        """Get recommendations for improving dataset quality"""
        recommendations = []
        
        for dataset_key, result in validation_results.items():
            if dataset_key.startswith('_') or not isinstance(result, dict):
                continue
            
            if not result.get('passed', False):
                recommendations.append(f"Dataset '{dataset_key}' failed validation (Score: {result.get('overall_quality_score', 0)}/100)")
                
                # Add specific recommendations based on failed checks
                checks = result.get('checks', {})
                for check_name, check_result in checks.items():
                    if not check_result.get('passed', False):
                        recommendations.append(f"  - {check_name}: {check_result.get('message', 'Check failed')}")
        
        if not recommendations:
            recommendations.append("All datasets passed validation successfully!")
        
        return recommendations