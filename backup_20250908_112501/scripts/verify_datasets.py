#!/usr/bin/env python3
"""
Real Dataset Verification Script for Persian Legal AI
اسکریپت تأیید واقعی مجموعه داده‌ها برای هوش مصنوعی حقوقی فارسی

This script verifies the authenticity and quality of Persian legal datasets
from verified sources like Hugging Face. NO MOCK DATA OR FAKE VALIDATION.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

try:
    from datasets import load_dataset, Dataset
    import huggingface_hub
    from huggingface_hub import HfApi, list_datasets
except ImportError as e:
    print(f"❌ Required dependencies not installed: {e}")
    print("Please install: pip install datasets huggingface_hub")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersianLegalDatasetVerifier:
    """Verifies authenticity and quality of Persian legal datasets"""
    
    VERIFIED_DATASETS = {
        'lscp_legal': {
            'name': 'LSCP Persian Legal Corpus',
            'hf_path': 'lscp/legal-persian',
            'type': 'comprehensive',
            'verified': True,
            'min_samples': 1000,
            'expected_size_mb': 2000
        },
        'hooshvare_legal': {
            'name': 'HooshvareLab Legal Persian',
            'hf_path': 'HooshvareLab/legal-persian', 
            'type': 'general_legal',
            'verified': True,
            'min_samples': 500,
            'expected_size_mb': 1500
        },
        'judicial_rulings': {
            'name': 'Persian Judicial Rulings',
            'hf_path': 'persian-judicial-rulings',
            'type': 'case_law',
            'verified': True,
            'min_samples': 200,
            'expected_size_mb': 1200
        }
    }
    
    def __init__(self):
        self.hf_api = HfApi()
        self.verification_results = {}
    
    def verify_all_datasets(self) -> Dict[str, bool]:
        """Verify all configured datasets"""
        logger.info("🔍 Starting dataset verification process...")
        
        for dataset_key, config in self.VERIFIED_DATASETS.items():
            logger.info(f"Verifying {config['name']}...")
            try:
                is_valid = self._verify_single_dataset(dataset_key, config)
                self.verification_results[dataset_key] = is_valid
                
                status = "✅ PASS" if is_valid else "❌ FAIL"
                logger.info(f"{status} - {config['name']}")
                
            except Exception as e:
                logger.error(f"❌ FAIL - {config['name']}: {str(e)}")
                self.verification_results[dataset_key] = False
        
        return self.verification_results
    
    def _verify_single_dataset(self, dataset_key: str, config: Dict[str, Any]) -> bool:
        """Verify a single dataset with real validation checks"""
        try:
            # Check if dataset exists on Hugging Face
            if not self._check_dataset_exists(config['hf_path']):
                logger.error(f"Dataset {config['hf_path']} not found on Hugging Face")
                return False
            
            # Load dataset for validation
            logger.info(f"Loading dataset {config['hf_path']} for validation...")
            dataset = load_dataset(config['hf_path'], split='train', streaming=True)
            
            # Perform real validation checks
            validation_checks = [
                self._check_minimum_samples(dataset, config['min_samples']),
                self._check_persian_content_quality(dataset),
                self._check_legal_relevance(dataset),
                self._check_data_format_consistency(dataset)
            ]
            
            all_passed = all(validation_checks)
            
            if all_passed:
                logger.info(f"✅ All validation checks passed for {config['name']}")
            else:
                logger.warning(f"⚠️ Some validation checks failed for {config['name']}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Error verifying dataset {dataset_key}: {str(e)}")
            return False
    
    def _check_dataset_exists(self, hf_path: str) -> bool:
        """Check if dataset exists on Hugging Face Hub"""
        try:
            # Use Hugging Face API to check if dataset exists
            datasets = list_datasets(search=hf_path)
            return any(dataset.id == hf_path for dataset in datasets)
        except Exception as e:
            logger.error(f"Error checking dataset existence: {e}")
            return False
    
    def _check_minimum_samples(self, dataset, min_samples: int) -> bool:
        """Check if dataset has minimum required samples"""
        try:
            # Count samples in streaming dataset
            sample_count = 0
            for _ in dataset.take(min_samples + 100):  # Take a bit more to be sure
                sample_count += 1
                if sample_count >= min_samples:
                    break
            
            logger.info(f"Dataset has {sample_count} samples (minimum required: {min_samples})")
            return sample_count >= min_samples
            
        except Exception as e:
            logger.error(f"Error checking sample count: {e}")
            return False
    
    def _check_persian_content_quality(self, dataset) -> bool:
        """Check if dataset contains quality Persian content"""
        try:
            persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
            sample_count = 0
            persian_content_count = 0
            
            for sample in dataset.take(100):  # Check first 100 samples
                sample_count += 1
                
                # Get text content from sample
                text_content = ""
                if isinstance(sample, dict):
                    # Find text field
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 50:
                            text_content = value
                            break
                elif isinstance(sample, str):
                    text_content = sample
                
                if text_content:
                    # Check if text contains Persian characters
                    persian_char_count = sum(1 for char in text_content if char in persian_chars)
                    if persian_char_count > len(text_content) * 0.3:  # At least 30% Persian
                        persian_content_count += 1
            
            persian_ratio = persian_content_count / sample_count if sample_count > 0 else 0
            logger.info(f"Persian content ratio: {persian_ratio:.2%}")
            
            return persian_ratio >= 0.5  # At least 50% Persian content
            
        except Exception as e:
            logger.error(f"Error checking Persian content: {e}")
            return False
    
    def _check_legal_relevance(self, dataset) -> bool:
        """Check if dataset contains legal-relevant content"""
        try:
            legal_keywords = [
                'قانون', 'حقوق', 'دادگاه', 'قاضی', 'وکیل', 'محکمه', 'حکم', 'رأی',
                'ماده', 'بند', 'تبصره', 'مقررات', 'آیین‌نامه', 'دستورالعمل',
                'شکایت', 'دعوا', 'محاکمه', 'محکومیت', 'برائت', 'جریمه'
            ]
            
            sample_count = 0
            legal_content_count = 0
            
            for sample in dataset.take(50):  # Check first 50 samples
                sample_count += 1
                
                # Get text content
                text_content = ""
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        if isinstance(value, str):
                            text_content += value + " "
                elif isinstance(sample, str):
                    text_content = sample
                
                # Check for legal keywords
                legal_keyword_count = sum(1 for keyword in legal_keywords if keyword in text_content)
                if legal_keyword_count >= 2:  # At least 2 legal keywords
                    legal_content_count += 1
            
            legal_ratio = legal_content_count / sample_count if sample_count > 0 else 0
            logger.info(f"Legal relevance ratio: {legal_ratio:.2%}")
            
            return legal_ratio >= 0.3  # At least 30% legal content
            
        except Exception as e:
            logger.error(f"Error checking legal relevance: {e}")
            return False
    
    def _check_data_format_consistency(self, dataset) -> bool:
        """Check if dataset has consistent format"""
        try:
            sample_count = 0
            consistent_format_count = 0
            
            for sample in dataset.take(20):  # Check first 20 samples
                sample_count += 1
                
                # Check if sample has consistent structure
                if isinstance(sample, dict) and len(sample) > 0:
                    # Check if all values are strings or have text content
                    has_text_content = any(
                        isinstance(value, str) and len(value.strip()) > 10 
                        for value in sample.values()
                    )
                    if has_text_content:
                        consistent_format_count += 1
                elif isinstance(sample, str) and len(sample.strip()) > 10:
                    consistent_format_count += 1
            
            consistency_ratio = consistent_format_count / sample_count if sample_count > 0 else 0
            logger.info(f"Format consistency ratio: {consistency_ratio:.2%}")
            
            return consistency_ratio >= 0.8  # At least 80% consistent format
            
        except Exception as e:
            logger.error(f"Error checking format consistency: {e}")
            return False
    
    def generate_verification_report(self) -> str:
        """Generate a detailed verification report"""
        report = []
        report.append("📊 Persian Legal Dataset Verification Report")
        report.append("=" * 50)
        
        total_datasets = len(self.verification_results)
        passed_datasets = sum(1 for result in self.verification_results.values() if result)
        
        report.append(f"Total Datasets: {total_datasets}")
        report.append(f"Passed Verification: {passed_datasets}")
        report.append(f"Failed Verification: {total_datasets - passed_datasets}")
        report.append("")
        
        for dataset_key, is_valid in self.verification_results.items():
            config = self.VERIFIED_DATASETS[dataset_key]
            status = "✅ PASS" if is_valid else "❌ FAIL"
            report.append(f"{status} - {config['name']}")
            report.append(f"   Path: {config['hf_path']}")
            report.append(f"   Type: {config['type']}")
            report.append("")
        
        if passed_datasets == total_datasets:
            report.append("🎉 All datasets verified successfully!")
            report.append("Ready for integration with Persian Legal AI system.")
        else:
            report.append("⚠️ Some datasets failed verification.")
            report.append("Please check the logs for details.")
        
        return "\n".join(report)

def main():
    """Main verification function"""
    print("🔍 Persian Legal Dataset Verification")
    print("=" * 40)
    
    try:
        verifier = PersianLegalDatasetVerifier()
        results = verifier.verify_all_datasets()
        
        print("\n" + verifier.generate_verification_report())
        
        # Return success if all datasets passed
        all_passed = all(results.values())
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Verification process failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)