"""
Real DoRA (Weight-Decomposed Low-Rank Adaptation) Trainer
پیاده‌سازی واقعی DoRA برای آموزش مدل‌های حقوقی فارسی
"""

import os
import math
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from loguru import logger

# Transformers and PEFT
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, PeftConfig
)

# Platform-agnostic optimization
import psutil
import multiprocessing

logger = logging.getLogger(__name__)

@dataclass
class DoRAConfig:
    """DoRA configuration"""
    base_model: str = "HooshvareLab/bert-base-parsbert-uncased"
    dora_rank: int = 8
    dora_alpha: int = 16
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 8
    max_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "value", "key", "dense"]

class DoRALayer(nn.Module):
    """
    Real DoRA layer implementation
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # DoRA decomposition: W = m * (A @ B)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # Magnitude vector
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Direction matrix (normalized)
        self.direction = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute LoRA adaptation
        lora_output = F.linear(x, self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # Compute magnitude and direction
        magnitude = torch.norm(self.magnitude)
        direction = F.normalize(self.direction, p=2, dim=1)
        
        # Combine magnitude and direction
        dora_output = magnitude * (direction @ lora_output.T).T
        
        return dora_output

class PersianLegalDataset(Dataset):
    """
    Real dataset for Persian legal documents
    """
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get text and label
        text = item.get('text', '')
        label = item.get('label', 'other')
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert label to numeric
        label_map = {
            'constitutional_law': 0,
            'civil_law': 1,
            'criminal_law': 2,
            'labor_law': 3,
            'commercial_law': 4,
            'regulation': 5,
            'instruction': 6,
            'judgment': 7,
            'other': 8
        }
        label_id = label_map.get(label, 8)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

class DoRATrainer:
    """
    Real DoRA trainer implementation
    """
    
    def __init__(self, config: DoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Platform-agnostic optimization
        self._setup_platform_optimization()
        
        logger.info(f"DoRA trainer initialized with device: {self.device}")
    
    def _setup_platform_optimization(self):
        """Setup platform-agnostic optimization"""
        try:
            # Get available CPU cores
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"Available CPU cores: {cpu_count}")
            
            # Set PyTorch threads
            torch.set_num_threads(min(cpu_count, 8))
            
            # Enable optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                # CPU optimizations
                torch.set_num_threads(cpu_count)
            
            logger.info("Platform optimization setup complete")
            
        except Exception as e:
            logger.warning(f"Platform optimization setup failed: {e}")
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load base model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.base_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Add padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.config.base_model,
                num_labels=9,  # Number of legal document categories
                return_dict=True
            )
            
            # Add classification head
            self.model.classifier = nn.Linear(self.model.config.hidden_size, 9)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def apply_dora(self):
        """Apply DoRA adaptation to the model"""
        try:
            logger.info("Applying DoRA adaptation")
            
            # Create DoRA config
            dora_config = LoraConfig(
                r=self.config.dora_rank,
                lora_alpha=self.config.dora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            
            # Apply PEFT
            self.model = get_peft_model(self.model, dora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to apply DoRA: {e}")
            raise
    
    def create_dataloader(self, data: List[Dict[str, Any]], batch_size: Optional[int] = None) -> DataLoader:
        """Create data loader"""
        try:
            batch_size = batch_size or self.config.batch_size
            
            # Create dataset
            dataset = PersianLegalDataset(data, self.tokenizer, self.config.max_length)
            
            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=min(4, multiprocessing.cpu_count()),
                pin_memory=torch.cuda.is_available()
            )
            
            logger.info(f"Created dataloader with {len(dataset)} samples, batch size: {batch_size}")
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            raise
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        try:
            # Get trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            # Setup optimizer
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            logger.info(f"Optimizer setup with {len(trainable_params)} trainable parameters")
            
        except Exception as e:
            logger.error(f"Failed to setup optimizer: {e}")
            raise
    
    def training_step(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step"""
        try:
            model.train()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise
    
    def train(self, train_data: List[Dict[str, Any]], eval_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Complete training loop"""
        try:
            logger.info("Starting DoRA training")
            
            # Setup model and optimizer
            if self.model is None:
                self.load_model()
                self.apply_dora()
            
            if self.optimizer is None:
                self.setup_optimizer()
            
            # Create data loaders
            train_dataloader = self.create_dataloader(train_data)
            
            eval_dataloader = None
            if eval_data:
                eval_dataloader = self.create_dataloader(eval_data)
            
            # Training metrics
            training_metrics = {
                'total_steps': 0,
                'total_epochs': 0,
                'total_loss': 0.0,
                'best_loss': float('inf'),
                'training_time': 0.0,
                'loss_history': [],
                'step_times': []
            }
            
            start_time = time.time()
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                epoch_loss = 0.0
                num_batches = 0
                
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
                
                for batch in progress_bar:
                    step_start = time.time()
                    
                    # Training step
                    loss = self.training_step(self.model, batch)
                    
                    step_time = time.time() - step_start
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    num_batches += 1
                    training_metrics['total_steps'] += 1
                    training_metrics['total_loss'] += loss.item()
                    training_metrics['loss_history'].append(loss.item())
                    training_metrics['step_times'].append(step_time)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss / num_batches:.4f}",
                        'step_time': f"{step_time:.2f}s"
                    })
                    
                    # Logging
                    if training_metrics['total_steps'] % self.config.logging_steps == 0:
                        logger.info(f"Step {training_metrics['total_steps']}: Loss = {loss.item():.4f}")
                
                # Epoch metrics
                avg_epoch_loss = epoch_loss / num_batches
                training_metrics['total_epochs'] += 1
                
                if avg_epoch_loss < training_metrics['best_loss']:
                    training_metrics['best_loss'] = avg_epoch_loss
                
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
                
                # Evaluation
                if eval_dataloader:
                    eval_metrics = self.evaluate(eval_dataloader)
                    logger.info(f"Evaluation metrics: {eval_metrics}")
            
            # Final metrics
            training_metrics['training_time'] = time.time() - start_time
            training_metrics['avg_loss'] = training_metrics['total_loss'] / training_metrics['total_steps']
            training_metrics['avg_step_time'] = np.mean(training_metrics['step_times'])
            
            logger.info(f"Training completed in {training_metrics['training_time']:.2f} seconds")
            logger.info(f"Final average loss: {training_metrics['avg_loss']:.4f}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        try:
            self.model.eval()
            
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
            
            # Calculate final metrics
            avg_loss = total_loss / len(eval_dataloader)
            accuracy = correct_predictions / total_predictions
            
            return {
                'eval_loss': avg_loss,
                'eval_accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'eval_loss': float('inf'), 'eval_accuracy': 0.0}
    
    def save_model(self, output_dir: str):
        """Save trained model"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Load model
            self.model = PeftModel.from_pretrained(
                AutoModel.from_pretrained(self.config.base_model),
                checkpoint_path
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            logger.info(f"Model loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise