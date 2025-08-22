"""
QR-Adaptor: Joint Bit-width and Rank Optimization
Advanced 2025 implementation with adaptive quantization and CPU optimization
"""

import os
import math
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger

# Quantization and optimization
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# Transformers
from transformers import AutoModel, AutoTokenizer

# Local imports
from config.training_config import QRAdaptorConfig, get_config


class NF4Quantizer:
    """
    NormalFloat 4-bit quantizer with double quantization
    Implements the NF4 quantization scheme for optimal CPU performance
    """
    
    def __init__(self, double_quantization: bool = True):
        self.double_quantization = double_quantization
        
        # NF4 quantization levels (normalized)
        self.nf4_levels = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ], dtype=torch.float32)
        
        logger.info(f"NF4 quantizer initialized with double_quantization={double_quantization}")
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NF4 format
        Returns: (quantized_tensor, scale, zero_point)
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Compute scale and zero point
        tensor_min = tensor_flat.min()
        tensor_max = tensor_flat.max()
        
        scale = (tensor_max - tensor_min) / 15.0  # 4-bit has 16 levels
        zero_point = tensor_min
        
        # Normalize tensor
        normalized = (tensor_flat - zero_point) / (scale + 1e-8)
        
        # Find closest NF4 levels
        distances = torch.abs(normalized.unsqueeze(1) - self.nf4_levels.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)
        
        # Create quantized tensor
        quantized = self.nf4_levels[indices].reshape(original_shape)
        
        # Double quantization for scale if enabled
        if self.double_quantization and scale.numel() > 1:
            scale_quantized, scale_scale, scale_zero = self._quantize_scale(scale)
            return quantized, (scale_quantized, scale_scale, scale_zero), zero_point
        
        return quantized, scale, zero_point
    
    def dequantize(self, quantized: torch.Tensor, scale: Union[torch.Tensor, Tuple], zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize NF4 tensor back to float"""
        
        # Handle double quantization
        if isinstance(scale, tuple):
            scale_quantized, scale_scale, scale_zero = scale
            scale = scale_quantized * scale_scale + scale_zero
        
        # Dequantize
        dequantized = quantized * scale + zero_point
        return dequantized
    
    def _quantize_scale(self, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply double quantization to scale values"""
        scale_min = scale.min()
        scale_max = scale.max()
        
        scale_scale = (scale_max - scale_min) / 255.0  # 8-bit for scale
        scale_zero = scale_min
        
        scale_normalized = (scale - scale_zero) / (scale_scale + 1e-8)
        scale_quantized = torch.round(scale_normalized).clamp(0, 255)
        
        return scale_quantized, scale_scale, scale_zero


class AdaptiveRankController:
    """
    Adaptive rank controller for dynamic rank adjustment
    Based on gradient sensitivity and importance scoring
    """
    
    def __init__(
        self,
        min_rank: int = 8,
        max_rank: int = 128,
        adjustment_frequency: int = 100,
        importance_threshold: float = 0.01
    ):
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.adjustment_frequency = adjustment_frequency
        self.importance_threshold = importance_threshold
        
        # Tracking metrics
        self.gradient_history = []
        self.importance_scores = {}
        self.rank_history = []
        self.adjustment_count = 0
        
        logger.info(f"Adaptive rank controller initialized: range=[{min_rank}, {max_rank}]")
    
    def compute_importance_score(self, layer_name: str, gradients: torch.Tensor) -> float:
        """Compute importance score based on gradient magnitudes"""
        
        # Compute gradient norm
        grad_norm = torch.norm(gradients).item()
        
        # Update running average
        if layer_name not in self.importance_scores:
            self.importance_scores[layer_name] = grad_norm
        else:
            # Exponential moving average
            alpha = 0.1
            self.importance_scores[layer_name] = (
                alpha * grad_norm + (1 - alpha) * self.importance_scores[layer_name]
            )
        
        return self.importance_scores[layer_name]
    
    def should_adjust_rank(self, step: int) -> bool:
        """Check if rank should be adjusted at this step"""
        return step > 0 and step % self.adjustment_frequency == 0
    
    def suggest_rank_adjustment(self, layer_name: str, current_rank: int) -> int:
        """Suggest new rank based on importance score"""
        
        if layer_name not in self.importance_scores:
            return current_rank
        
        importance = self.importance_scores[layer_name]
        
        # Normalize importance score
        all_scores = list(self.importance_scores.values())
        if len(all_scores) > 1:
            mean_importance = np.mean(all_scores)
            std_importance = np.std(all_scores) + 1e-8
            normalized_importance = (importance - mean_importance) / std_importance
        else:
            normalized_importance = 0.0
        
        # Adjust rank based on normalized importance
        if normalized_importance > 1.0:  # High importance
            new_rank = min(self.max_rank, int(current_rank * 1.2))
        elif normalized_importance < -1.0:  # Low importance
            new_rank = max(self.min_rank, int(current_rank * 0.8))
        else:
            new_rank = current_rank
        
        # Record adjustment
        if new_rank != current_rank:
            self.adjustment_count += 1
            logger.debug(f"Rank adjustment for {layer_name}: {current_rank} -> {new_rank}")
        
        return new_rank
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics"""
        return {
            'total_adjustments': self.adjustment_count,
            'importance_scores': self.importance_scores.copy(),
            'rank_range': (self.min_rank, self.max_rank),
            'avg_importance': np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.0
        }


class QRAdaptorLayer(nn.Module):
    """
    QR-Adaptor layer with joint bit-width and rank optimization
    Implements adaptive quantization and rank adjustment
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        config: QRAdaptorConfig,
        layer_name: str,
        quantizer: NF4Quantizer,
        rank_controller: AdaptiveRankController
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.config = config
        self.layer_name = layer_name
        self.quantizer = quantizer
        self.rank_controller = rank_controller
        
        # Get layer dimensions
        if hasattr(original_layer, 'weight'):
            self.weight_shape = original_layer.weight.shape
            self.in_features = self.weight_shape[1]
            self.out_features = self.weight_shape[0]
        else:
            raise ValueError("Original layer must have weight attribute")
        
        # Initialize adaptive parameters
        self.current_rank = self._determine_initial_rank()
        self.current_bits = self._determine_initial_bits()
        
        # Initialize adaptor components
        self._initialize_adaptor_components()
        
        # Training metrics
        self.metrics = {
            'rank_history': [self.current_rank],
            'bits_history': [self.current_bits],
            'quantization_error': [],
            'compression_ratio': [],
            'inference_time': []
        }
        
        logger.info(f"QR-Adaptor layer {layer_name} initialized: rank={self.current_rank}, bits={self.current_bits}")
    
    def _determine_initial_rank(self) -> int:
        """Determine initial rank based on layer importance"""
        # Use SVD to estimate optimal rank
        with torch.no_grad():
            weight = self.original_layer.weight
            U, S, V = torch.svd(weight)
            
            # Find rank that captures 95% of energy
            total_energy = torch.sum(S ** 2)
            cumsum_energy = torch.cumsum(S ** 2, dim=0)
            energy_ratio = cumsum_energy / total_energy
            
            optimal_rank = torch.argmax((energy_ratio >= 0.95).float()).item() + 1
            
            # Clamp to allowed range
            return max(self.rank_controller.min_rank, 
                      min(self.rank_controller.max_rank, optimal_rank))
    
    def _determine_initial_bits(self) -> int:
        """Determine initial bit-width based on layer type"""
        # Classify layer importance
        if any(critical in self.layer_name for critical in ['attention', 'query', 'key', 'value']):
            return self.config.quantization_bits['critical_layers']
        elif any(important in self.layer_name for important in ['ffn', 'mlp', 'dense']):
            return self.config.quantization_bits['standard_layers']
        else:
            return self.config.quantization_bits['less_critical_layers']
    
    def _initialize_adaptor_components(self) -> None:
        """Initialize QR-Adaptor components"""
        
        # Low-rank adaptation matrices
        self.adaptor_A = nn.Parameter(
            torch.randn(self.current_rank, self.in_features) * 0.01
        )
        self.adaptor_B = nn.Parameter(
            torch.randn(self.out_features, self.current_rank) * 0.01
        )
        
        # Quantization parameters
        self.quantized_weight = None
        self.quantization_scale = None
        self.quantization_zero_point = None
        
        # Initialize quantized version
        self._update_quantization()
    
    def _update_quantization(self) -> None:
        """Update quantized weight representation"""
        with torch.no_grad():
            # Get current adaptor weight
            adaptor_weight = self.adaptor_B @ self.adaptor_A
            
            # Quantize based on current bit-width
            if self.current_bits == 4:
                quantized, scale, zero_point = self.quantizer.quantize(adaptor_weight)
                self.quantized_weight = quantized
                self.quantization_scale = scale
                self.quantization_zero_point = zero_point
            elif self.current_bits == 8:
                # 8-bit quantization (simpler)
                weight_min = adaptor_weight.min()
                weight_max = adaptor_weight.max()
                scale = (weight_max - weight_min) / 255.0
                zero_point = weight_min
                
                normalized = (adaptor_weight - zero_point) / (scale + 1e-8)
                quantized = torch.round(normalized).clamp(0, 255)
                
                self.quantized_weight = quantized
                self.quantization_scale = scale
                self.quantization_zero_point = zero_point
            else:
                # 16-bit or higher - no quantization
                self.quantized_weight = adaptor_weight
                self.quantization_scale = torch.tensor(1.0)
                self.quantization_zero_point = torch.tensor(0.0)
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight for forward pass"""
        if self.current_bits <= 4:
            return self.quantizer.dequantize(
                self.quantized_weight,
                self.quantization_scale,
                self.quantization_zero_point
            )
        elif self.current_bits == 8:
            return self.quantized_weight * self.quantization_scale + self.quantization_zero_point
        else:
            return self.quantized_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized adaptation"""
        
        # Original layer output
        original_output = self.original_layer(x)
        
        # Adaptor contribution
        if self.training:
            # During training, use full precision
            adaptor_output = F.linear(x, self.adaptor_B @ self.adaptor_A)
        else:
            # During inference, use quantized weights
            adaptor_weight = self._dequantize_weight()
            adaptor_output = F.linear(x, adaptor_weight)
        
        return original_output + adaptor_output
    
    def update_rank(self, new_rank: int) -> None:
        """Update adaptor rank dynamically"""
        if new_rank == self.current_rank:
            return
        
        old_rank = self.current_rank
        self.current_rank = new_rank
        
        # Resize adaptor matrices
        with torch.no_grad():
            if new_rank > old_rank:
                # Expand matrices
                new_A = torch.zeros(new_rank, self.in_features)
                new_B = torch.zeros(self.out_features, new_rank)
                
                new_A[:old_rank] = self.adaptor_A.data
                new_B[:, :old_rank] = self.adaptor_B.data
                
                # Initialize new parameters
                new_A[old_rank:] = torch.randn(new_rank - old_rank, self.in_features) * 0.01
                new_B[:, old_rank:] = torch.randn(self.out_features, new_rank - old_rank) * 0.01
            else:
                # Shrink matrices (keep most important components)
                new_A = self.adaptor_A.data[:new_rank]
                new_B = self.adaptor_B.data[:, :new_rank]
            
            # Update parameters
            self.adaptor_A = nn.Parameter(new_A)
            self.adaptor_B = nn.Parameter(new_B)
        
        # Update quantization
        self._update_quantization()
        
        # Record change
        self.metrics['rank_history'].append(new_rank)
        logger.debug(f"Layer {self.layer_name} rank updated: {old_rank} -> {new_rank}")
    
    def update_quantization_bits(self, new_bits: int) -> None:
        """Update quantization bit-width"""
        if new_bits == self.current_bits:
            return
        
        old_bits = self.current_bits
        self.current_bits = new_bits
        
        # Update quantization
        self._update_quantization()
        
        # Record change
        self.metrics['bits_history'].append(new_bits)
        logger.debug(f"Layer {self.layer_name} bits updated: {old_bits} -> {new_bits}")
    
    def get_compression_ratio(self) -> float:
        """Calculate current compression ratio"""
        original_params = self.out_features * self.in_features
        
        # Adaptor parameters
        adaptor_params = self.current_rank * (self.in_features + self.out_features)
        
        # Quantization factor
        quantization_factor = 32 / self.current_bits  # Assuming original is 32-bit
        
        compressed_size = adaptor_params / quantization_factor
        compression_ratio = original_params / compressed_size
        
        return compression_ratio
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get layer metrics"""
        return {
            'current_rank': self.current_rank,
            'current_bits': self.current_bits,
            'compression_ratio': self.get_compression_ratio(),
            'parameter_count': self.current_rank * (self.in_features + self.out_features),
            'metrics_history': self.metrics.copy()
        }


class QRAdaptor:
    """
    Main QR-Adaptor system with joint optimization
    Manages multiple QR-Adaptor layers with adaptive control
    """
    
    def __init__(
        self,
        base_model: str,
        quantization_bits: Dict[str, int] = None,
        adaptive_rank: bool = True,
        optimization_target: str = "cpu"
    ):
        self.base_model_name = base_model
        self.config = QRAdaptorConfig()
        
        if quantization_bits:
            self.config.quantization_bits.update(quantization_bits)
        
        self.adaptive_rank = adaptive_rank
        self.optimization_target = optimization_target
        
        # Initialize components
        self.quantizer = NF4Quantizer(double_quantization=self.config.double_quantization)
        self.rank_controller = AdaptiveRankController(
            min_rank=self.config.min_rank,
            max_rank=self.config.max_rank,
            adjustment_frequency=self.config.rank_adjustment_frequency,
            importance_threshold=self.config.importance_threshold
        )
        
        # Model components
        self.base_model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.qr_layers: Dict[str, QRAdaptorLayer] = {}
        
        # Training state
        self.training_active = False
        self.global_step = 0
        self.optimization_history = {
            'compression_ratios': [],
            'inference_times': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
        
        # Loaded models from main system
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"QR-Adaptor initialized for model: {base_model}")
    
    async def initialize_model(self) -> bool:
        """Initialize base model and apply QR-Adaptor layers"""
        try:
            # Use pre-loaded model if available
            if self.base_model_name in self.loaded_models:
                logger.info(f"Using pre-loaded model: {self.base_model_name}")
                self.tokenizer = self.loaded_models[self.base_model_name]['tokenizer']
                self.base_model = self.loaded_models[self.base_model_name]['model']
            else:
                # Load model and tokenizer
                logger.info(f"Loading model: {self.base_model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                    cache_dir="./models/cache"
                )
                
                self.base_model = AutoModel.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    cache_dir="./models/cache"
                )
            
            # Apply QR-Adaptor layers
            self._apply_qr_layers()
            
            # Apply CPU optimizations
            if self.optimization_target == "cpu" and IPEX_AVAILABLE:
                logger.info("Applying Intel Extension optimizations")
                self.base_model = ipex.optimize(self.base_model, level="O1")
            
            logger.success("QR-Adaptor model initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"QR-Adaptor initialization failed: {e}")
            return False
    
    def _apply_qr_layers(self) -> None:
        """Apply QR-Adaptor layers to target modules"""
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Create QR-Adaptor layer
                    qr_layer = QRAdaptorLayer(
                        original_layer=module,
                        config=self.config,
                        layer_name=name,
                        quantizer=self.quantizer,
                        rank_controller=self.rank_controller
                    )
                    
                    # Replace the module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent_module = dict(self.base_model.named_modules())[parent_name]
                        setattr(parent_module, child_name, qr_layer)
                    else:
                        setattr(self.base_model, child_name, qr_layer)
                    
                    self.qr_layers[name] = qr_layer
                    
                    logger.debug(f"Applied QR-Adaptor to layer: {name}")
        
        logger.info(f"Applied QR-Adaptor to {len(self.qr_layers)} layers")
    
    async def start_optimization(self, training_params: Dict[str, Any]) -> bool:
        """Start QR-Adaptor optimization process"""
        
        if self.training_active:
            logger.warning("Optimization already active")
            return False
        
        try:
            logger.info("Starting QR-Adaptor optimization...")
            self.training_active = True
            
            # Initialize model if not done
            if self.base_model is None:
                if not await self.initialize_model():
                    return False
            
            # Run optimization loop
            num_steps = training_params.get('optimization_steps', 1000)
            await self._run_optimization_loop(num_steps, training_params)
            
            logger.success("QR-Adaptor optimization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"QR-Adaptor optimization failed: {e}")
            return False
        finally:
            self.training_active = False
    
    async def _run_optimization_loop(self, num_steps: int, params: Dict[str, Any]) -> None:
        """Run the joint optimization loop"""
        
        pbar = tqdm(range(num_steps), desc="QR-Adaptor Optimization")
        
        for step in pbar:
            self.global_step = step
            
            # Simulate optimization step
            await self._optimization_step()
            
            # Adaptive rank adjustment
            if self.adaptive_rank and self.rank_controller.should_adjust_rank(step):
                await self._adjust_ranks()
            
            # Joint bit-width optimization
            if step % 200 == 0:
                await self._optimize_bit_widths()
            
            # Update progress
            avg_compression = np.mean([
                layer.get_compression_ratio() for layer in self.qr_layers.values()
            ])
            
            pbar.set_postfix({
                'avg_compression': f"{avg_compression:.2f}x",
                'active_layers': len(self.qr_layers)
            })
            
            # Log metrics periodically
            if step % 50 == 0:
                await self._log_optimization_metrics()
            
            # Small delay
            await asyncio.sleep(0.01)
    
    async def _optimization_step(self) -> None:
        """Simulate single optimization step"""
        
        # In real implementation, this would:
        # 1. Forward pass with current quantization
        # 2. Compute reconstruction loss
        # 3. Update adaptor parameters
        # 4. Update quantization parameters
        
        # Simulate gradient computation for importance scoring
        for name, layer in self.qr_layers.items():
            # Simulate gradients
            fake_grad_A = torch.randn_like(layer.adaptor_A) * 0.01
            fake_grad_B = torch.randn_like(layer.adaptor_B) * 0.01
            
            # Compute importance score
            combined_grad = torch.cat([fake_grad_A.flatten(), fake_grad_B.flatten()])
            importance = self.rank_controller.compute_importance_score(name, combined_grad)
            
            # Update layer quantization periodically
            if self.global_step % 100 == 0:
                layer._update_quantization()
    
    async def _adjust_ranks(self) -> None:
        """Adjust ranks based on importance scores"""
        
        adjustments_made = 0
        
        for name, layer in self.qr_layers.items():
            current_rank = layer.current_rank
            suggested_rank = self.rank_controller.suggest_rank_adjustment(name, current_rank)
            
            if suggested_rank != current_rank:
                layer.update_rank(suggested_rank)
                adjustments_made += 1
        
        if adjustments_made > 0:
            logger.info(f"Adjusted ranks for {adjustments_made} layers")
    
    async def _optimize_bit_widths(self) -> None:
        """Optimize bit-widths based on joint optimization criteria"""
        
        # Compute joint optimization score for each layer
        for name, layer in self.qr_layers.items():
            current_bits = layer.current_bits
            current_compression = layer.get_compression_ratio()
            
            # Simple heuristic: increase bits for high-importance layers
            importance = self.rank_controller.importance_scores.get(name, 0.0)
            
            if importance > 0.1 and current_bits < 8:  # High importance
                new_bits = min(8, current_bits * 2)
            elif importance < 0.05 and current_bits > 4:  # Low importance
                new_bits = max(4, current_bits // 2)
            else:
                new_bits = current_bits
            
            if new_bits != current_bits:
                layer.update_quantization_bits(new_bits)
    
    async def _log_optimization_metrics(self) -> None:
        """Log comprehensive optimization metrics"""
        
        # Compute aggregate metrics
        total_compression = np.mean([
            layer.get_compression_ratio() for layer in self.qr_layers.values()
        ])
        
        avg_rank = np.mean([
            layer.current_rank for layer in self.qr_layers.values()
        ])
        
        avg_bits = np.mean([
            layer.current_bits for layer in self.qr_layers.values()
        ])
        
        # Store metrics
        self.optimization_history['compression_ratios'].append(total_compression)
        
        # Log key metrics
        if self.global_step % 100 == 0:
            logger.info(f"Step {self.global_step}: Compression={total_compression:.2f}x, "
                       f"Avg Rank={avg_rank:.1f}, Avg Bits={avg_bits:.1f}")
    
    async def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """Save QR-Adaptor checkpoint"""
        
        if checkpoint_path is None:
            checkpoint_dir = Path("./checkpoints/qr_adaptor")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        try:
            # Prepare checkpoint data
            checkpoint = {
                'model_name': self.base_model_name,
                'config': self.config.__dict__,
                'global_step': self.global_step,
                'qr_layers_state': {},
                'rank_controller_state': self.rank_controller.get_statistics(),
                'optimization_history': self.optimization_history
            }
            
            # Save QR layer states
            for name, layer in self.qr_layers.items():
                checkpoint['qr_layers_state'][name] = {
                    'adaptor_A': layer.adaptor_A.data,
                    'adaptor_B': layer.adaptor_B.data,
                    'current_rank': layer.current_rank,
                    'current_bits': layer.current_bits,
                    'quantized_weight': layer.quantized_weight,
                    'quantization_scale': layer.quantization_scale,
                    'quantization_zero_point': layer.quantization_zero_point,
                    'metrics': layer.get_metrics()
                }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"QR-Adaptor checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save QR-Adaptor checkpoint: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        layer_info = {}
        for name, layer in self.qr_layers.items():
            layer_info[name] = layer.get_metrics()
        
        return {
            'model_name': self.base_model_name,
            'config': self.config.__dict__,
            'training_active': self.training_active,
            'global_step': self.global_step,
            'total_layers': len(self.qr_layers),
            'rank_controller_stats': self.rank_controller.get_statistics(),
            'optimization_history': self.optimization_history,
            'layer_details': layer_info,
            'average_compression': np.mean([
                layer.get_compression_ratio() for layer in self.qr_layers.values()
            ]) if self.qr_layers else 0.0
        }
    
    def get_inference_model(self) -> nn.Module:
        """Get model optimized for inference"""
        
        if self.base_model is None:
            raise ValueError("Model not initialized")
        
        # Set all layers to evaluation mode for quantized inference
        self.base_model.eval()
        
        # Update all quantizations
        for layer in self.qr_layers.values():
            layer._update_quantization()
        
        logger.info("Model prepared for quantized inference")
        return self.base_model