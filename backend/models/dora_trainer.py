"""
DoRA (Weight-Decomposed Low-Rank Adaptation) Trainer
آموزش‌دهنده DoRA برای مدل‌های حقوقی فارسی
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from datetime import datetime
import json
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

logger = logging.getLogger(__name__)

class DoRALayer(nn.Module):
    """
    DoRA layer implementation with magnitude and direction decomposition
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 64, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Direction components (low-rank)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude component
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Scaling factor
        self.scaling = self.alpha / self.rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute direction: B @ A @ x
        direction = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        # Apply magnitude scaling
        output = direction * self.magnitude.unsqueeze(0) * self.scaling
        
        return output

class DoRATrainer:
    """
    Production-grade DoRA trainer for Persian legal models
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_name = model_config['base_model']
        self.rank = model_config.get('dora_rank', 64)
        self.alpha = model_config.get('dora_alpha', 16.0)
        self.target_modules = model_config.get('target_modules', 
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        self.model = None
        self.tokenizer = None
        self.dora_layers = {}
        self.optimizers = {}
        self.schedulers = {}
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Performance monitoring
        self.start_time = None
        self.memory_usage = []
        
    def load_model(self) -> Tuple[Any, Any]:
        """Load Persian model with Intel optimizations"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Apply Intel optimizations if available
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model, level="O1")
                logger.info("Applied Intel Extension optimizations")
            except ImportError:
                logger.warning("Intel Extension not available, using standard PyTorch")
            
            # Apply DoRA layers
            self._apply_dora_layers()
            
            logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_dora_layers(self):
        """Apply DoRA layers to target modules"""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Replace with DoRA layer
                    dora_layer = DoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.rank,
                        alpha=self.alpha
                    )
                    
                    # Copy original weights for initialization
                    with torch.no_grad():
                        dora_layer.lora_B.data = module.weight.data / self.alpha
                        dora_layer.magnitude.data = torch.norm(module.weight.data, dim=1)
                    
                    # Replace module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, dora_layer)
                    
                    self.dora_layers[name] = dora_layer
                    logger.info(f"Applied DoRA to {name}")
    
    def setup_optimizers(self, learning_rate: float = 1e-4, weight_decay: float = 0.01):
        """Setup separate optimizers for magnitude and direction components"""
        
        # Direction optimizer (LoRA parameters)
        direction_params = []
        for layer in self.dora_layers.values():
            direction_params.extend([layer.lora_A, layer.lora_B])
        
        self.optimizers['direction'] = AdamW(
            direction_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Magnitude optimizer
        magnitude_params = []
        for layer in self.dora_layers.values():
            magnitude_params.append(layer.magnitude)
        
        self.optimizers['magnitude'] = AdamW(
            magnitude_params,
            lr=learning_rate * 0.1,  # Lower learning rate for magnitude
            weight_decay=0.0,  # No weight decay for magnitude
            betas=(0.9, 0.999)
        )
        
        # Setup schedulers
        self.schedulers['direction'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizers['direction'], T_max=1000
        )
        self.schedulers['magnitude'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizers['magnitude'], T_max=1000
        )
        
        logger.info("Optimizers and schedulers setup complete")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with DoRA updates"""
        if self.start_time is None:
            self.start_time = datetime.now()
        
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Backward pass for direction components
        self.optimizers['direction'].zero_grad()
        loss.backward(retain_graph=True)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizers['direction'].param_groups[0]['params']],
            max_norm=1.0
        )
        
        self.optimizers['direction'].step()
        
        # Backward pass for magnitude components
        self.optimizers['magnitude'].zero_grad()
        loss.backward()
        
        # Gradient clipping for magnitude
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizers['magnitude'].param_groups[0]['params']],
            max_norm=1.0
        )
        
        self.optimizers['magnitude'].step()
        
        # Update schedulers
        self.schedulers['direction'].step()
        self.schedulers['magnitude'].step()
        
        # Update training state
        self.current_step += 1
        
        # Collect metrics
        metrics = {
            'loss': float(loss),
            'learning_rate_direction': self.optimizers['direction'].param_groups[0]['lr'],
            'learning_rate_magnitude': self.optimizers['magnitude'].param_groups[0]['lr'],
            'step': self.current_step,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage': psutil.cpu_percent()
        }
        
        self.training_history.append(metrics)
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = float(loss)
        
        return metrics
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_states': {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            'scheduler_states': {
                name: sched.state_dict() for name, sched in self.schedulers.items()
            },
            'metrics': metrics,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'rank': self.rank,
                'alpha': self.alpha,
                'target_modules': self.target_modules
            }
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics['loss'] == self.best_loss:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        for name, opt in self.optimizers.items():
            opt.load_state_dict(checkpoint['optimizer_states'][name])
        
        for name, sched in self.schedulers.items():
            sched.load_state_dict(checkpoint['scheduler_states'][name])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        if not self.training_history:
            return {}
        
        latest = self.training_history[-1]
        
        return {
            'current_loss': latest['loss'],
            'best_loss': self.best_loss,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'learning_rate_direction': latest['learning_rate_direction'],
            'learning_rate_magnitude': latest['learning_rate_magnitude'],
            'memory_usage_mb': latest['memory_usage_mb'],
            'cpu_usage': latest['cpu_usage'],
            'training_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def decompose_weights(self, weight_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose weight matrix into magnitude and direction"""
        # Compute magnitude (L2 norm of each output neuron)
        magnitude = torch.norm(weight_matrix, dim=1, keepdim=True)
        
        # Compute direction (normalized weights)
        direction = weight_matrix / (magnitude + 1e-8)
        
        return magnitude.squeeze(), direction
    
    def optimize_rank_allocation(self, model: nn.Module) -> Dict[str, int]:
        """Dynamically optimize rank allocation based on layer importance"""
        rank_allocation = {}
        
        for name, layer in self.dora_layers.items():
            # Compute gradient norm as importance measure
            if hasattr(layer, 'lora_A') and layer.lora_A.grad is not None:
                importance = torch.norm(layer.lora_A.grad).item()
                
                # Adjust rank based on importance
                if importance > 0.1:
                    rank_allocation[name] = min(self.rank * 2, 256)
                elif importance > 0.01:
                    rank_allocation[name] = self.rank
                else:
                    rank_allocation[name] = max(self.rank // 2, 16)
            else:
                rank_allocation[name] = self.rank
        
        return rank_allocation
    
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("DoRA trainer cleanup complete")