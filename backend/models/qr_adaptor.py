"""
QR-Adaptor: Joint Quantization and Rank Optimization
بهینه‌ساز ترکیبی کوانتیزاسیون و رنک برای مدل‌های فارسی
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Quantization types supported"""
    NF4 = "nf4"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    DYNAMIC = "dynamic"

@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    quant_type: QuantizationType = QuantizationType.NF4
    use_double_quantization: bool = True
    adaptive_bits: bool = True
    compression_target: float = 0.5  # Target compression ratio
    min_bits: int = 4
    max_bits: int = 16
    calibration_samples: int = 100

@dataclass
class RankConfig:
    """Configuration for rank optimization"""
    adaptive_rank: bool = True
    min_rank: int = 16
    max_rank: int = 256
    rank_step: int = 16
    importance_threshold: float = 0.01

class NF4Quantizer:
    """NF4 (NormalFloat4) quantizer implementation"""
    
    def __init__(self):
        # NF4 quantization constants
        self.nf4_constants = self._get_nf4_constants()
    
    def _get_nf4_constants(self) -> torch.Tensor:
        """Get NF4 quantization constants"""
        # NF4 values from QLoRA paper
        nf4_values = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
        ]
        return torch.tensor(nf4_values, dtype=torch.float32)
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to NF4"""
        # Normalize tensor to [-1, 1] range
        absmax = torch.max(torch.abs(tensor))
        normalized = tensor / (absmax + 1e-8)
        
        # Find closest NF4 values
        distances = torch.abs(normalized.unsqueeze(-1) - self.nf4_constants.unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        
        # Convert to 4-bit representation
        quantized = indices.to(torch.uint8)
        
        return quantized, absmax
    
    def dequantize(self, quantized: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
        """Dequantize NF4 tensor"""
        # Convert indices back to values
        values = self.nf4_constants[quantized]
        return values * absmax

class AdaptiveQuantizer:
    """Adaptive quantizer that selects optimal bit-width"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.nf4_quantizer = NF4Quantizer()
        self.bit_widths = [4, 8, 16]
        self.quantization_errors = {}
    
    def analyze_layer_importance(self, layer: nn.Module, sample_inputs: torch.Tensor) -> float:
        """Analyze layer importance for adaptive quantization"""
        with torch.no_grad():
            # Forward pass to get output
            output = layer(sample_inputs)
            
            # Compute gradient norm as importance measure
            if hasattr(layer, 'weight') and layer.weight.requires_grad:
                weight_norm = torch.norm(layer.weight).item()
                return weight_norm
            else:
                return 0.0
    
    def select_optimal_bits(self, layer: nn.Module, sample_inputs: torch.Tensor) -> int:
        """Select optimal bit-width for layer"""
        importance = self.analyze_layer_importance(layer, sample_inputs)
        
        # Select bit-width based on importance
        if importance > 0.1:
            return 16  # High importance - use full precision
        elif importance > 0.01:
            return 8   # Medium importance - use 8-bit
        else:
            return 4   # Low importance - use 4-bit
    
    def quantize_layer(self, layer: nn.Module, target_bits: int) -> Dict[str, Any]:
        """Quantize layer with specified bit-width"""
        if not hasattr(layer, 'weight'):
            return {'original': layer, 'quantized': layer, 'quantization_info': {}}
        
        weight = layer.weight.data
        
        if target_bits == 4:
            # Use NF4 quantization
            quantized_weight, absmax = self.nf4_quantizer.quantize(weight)
            quantization_info = {
                'type': 'nf4',
                'absmax': absmax,
                'compression_ratio': 4.0  # 32-bit to 4-bit
            }
        elif target_bits == 8:
            # Use INT8 quantization
            scale = 127.0 / torch.max(torch.abs(weight))
            quantized_weight = torch.round(weight * scale).clamp(-128, 127).to(torch.int8)
            quantization_info = {
                'type': 'int8',
                'scale': scale,
                'compression_ratio': 4.0  # 32-bit to 8-bit
            }
        else:
            # Use FP16
            quantized_weight = weight.half()
            quantization_info = {
                'type': 'fp16',
                'compression_ratio': 2.0  # 32-bit to 16-bit
            }
        
        # Create quantized layer
        quantized_layer = QuantizedLayer(layer, quantized_weight, quantization_info)
        
        return {
            'original': layer,
            'quantized': quantized_layer,
            'quantization_info': quantization_info
        }

class QuantizedLayer(nn.Module):
    """Wrapper for quantized layers"""
    
    def __init__(self, original_layer: nn.Module, quantized_weight: torch.Tensor, 
                 quantization_info: Dict[str, Any]):
        super().__init__()
        self.original_layer = original_layer
        self.quantized_weight = quantized_weight
        self.quantization_info = quantization_info
        self.quant_type = quantization_info['type']
        
        # Copy other attributes
        for attr_name in ['bias', 'in_features', 'out_features']:
            if hasattr(original_layer, attr_name):
                setattr(self, attr_name, getattr(original_layer, attr_name))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights"""
        if self.quant_type == 'nf4':
            # Dequantize NF4 weights
            absmax = self.quantization_info['absmax']
            weight = self.nf4_quantizer.dequantize(self.quantized_weight, absmax)
        elif self.quant_type == 'int8':
            # Dequantize INT8 weights
            scale = self.quantization_info['scale']
            weight = self.quantized_weight.float() / scale
        else:
            # FP16 weights
            weight = self.quantized_weight.float()
        
        return F.linear(x, weight, self.bias)

class RankOptimizer:
    """Dynamic rank optimization for LoRA layers"""
    
    def __init__(self, config: RankConfig):
        self.config = config
        self.rank_history = {}
        self.importance_scores = {}
    
    def compute_layer_importance(self, layer: nn.Module, gradients: torch.Tensor) -> float:
        """Compute importance score for rank optimization"""
        if gradients is None:
            return 0.0
        
        # Use gradient norm as importance measure
        importance = torch.norm(gradients).item()
        
        # Normalize by layer size
        if hasattr(layer, 'weight'):
            layer_size = layer.weight.numel()
            importance = importance / math.sqrt(layer_size)
        
        return importance
    
    def optimize_rank_allocation(self, model: nn.Module, gradients: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Optimize rank allocation based on layer importance"""
        rank_allocation = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Compute importance
                if name in gradients:
                    importance = self.compute_layer_importance(module, gradients[name])
                else:
                    importance = 0.0
                
                self.importance_scores[name] = importance
                
                # Determine optimal rank
                if importance > 0.1:
                    rank = min(self.config.max_rank, 
                              self.rank_history.get(name, self.config.min_rank) + self.config.rank_step)
                elif importance > 0.01:
                    rank = self.rank_history.get(name, self.config.min_rank)
                else:
                    rank = max(self.config.min_rank,
                              self.rank_history.get(name, self.config.min_rank) - self.config.rank_step)
                
                rank_allocation[name] = rank
                self.rank_history[name] = rank
        
        return rank_allocation
    
    def adjust_rank_dynamically(self, layer: nn.Module, target_rank: int) -> nn.Module:
        """Dynamically adjust rank of LoRA layer"""
        if not hasattr(layer, 'lora_A') or not hasattr(layer, 'lora_B'):
            return layer
        
        current_rank = layer.lora_A.shape[0]
        
        if target_rank == current_rank:
            return layer
        
        # Adjust rank
        if target_rank > current_rank:
            # Increase rank - pad with small random values
            pad_size = target_rank - current_rank
            new_A = torch.cat([
                layer.lora_A,
                torch.randn(pad_size, layer.lora_A.shape[1]) * 0.01
            ], dim=0)
            new_B = torch.cat([
                layer.lora_B,
                torch.zeros(layer.lora_B.shape[0], pad_size)
            ], dim=1)
        else:
            # Decrease rank - truncate
            new_A = layer.lora_A[:target_rank]
            new_B = layer.lora_B[:, :target_rank]
        
        # Update layer parameters
        layer.lora_A.data = new_A
        layer.lora_B.data = new_B
        
        return layer

class QRAdaptor:
    """
    Joint Quantization and Rank optimization for Persian models
    """
    
    def __init__(self, quantization_config: QuantizationConfig, rank_config: RankConfig):
        self.quantization_config = quantization_config
        self.rank_config = rank_config
        
        self.adaptive_quantizer = AdaptiveQuantizer(quantization_config)
        self.rank_optimizer = RankOptimizer(rank_config)
        
        self.quantization_stats = {}
        self.compression_ratios = {}
        
    def analyze_model_compression(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model compression potential"""
        total_params = sum(p.numel() for p in model.parameters())
        total_size_mb = total_params * 4 / (1024 * 1024)  # Assuming FP32
        
        compression_analysis = {
            'total_parameters': total_params,
            'original_size_mb': total_size_mb,
            'layers_analysis': {}
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                layer_params = module.weight.numel()
                layer_size_mb = layer_params * 4 / (1024 * 1024)
                
                # Analyze compression potential
                optimal_bits = self.adaptive_quantizer.select_optimal_bits(module, None)
                compressed_size_mb = layer_size_mb * (optimal_bits / 32)
                
                compression_analysis['layers_analysis'][name] = {
                    'parameters': layer_params,
                    'original_size_mb': layer_size_mb,
                    'optimal_bits': optimal_bits,
                    'compressed_size_mb': compressed_size_mb,
                    'compression_ratio': layer_size_mb / compressed_size_mb
                }
        
        return compression_analysis
    
    def apply_quantization(self, model: nn.Module, sample_inputs: Optional[torch.Tensor] = None) -> nn.Module:
        """Apply adaptive quantization to model"""
        quantized_layers = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d)):
                # Select optimal bit-width
                if sample_inputs is not None:
                    optimal_bits = self.adaptive_quantizer.select_optimal_bits(module, sample_inputs)
                else:
                    optimal_bits = 8  # Default to 8-bit
                
                # Quantize layer
                quantization_result = self.adaptive_quantizer.quantize_layer(module, optimal_bits)
                quantized_layers[name] = quantization_result
                
                # Update statistics
                self.quantization_stats[name] = {
                    'original_bits': 32,
                    'quantized_bits': optimal_bits,
                    'compression_ratio': 32 / optimal_bits
                }
        
        # Replace layers in model
        for name, result in quantized_layers.items():
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, result['quantized'])
            else:
                setattr(model, child_name, result['quantized'])
        
        logger.info(f"Applied quantization to {len(quantized_layers)} layers")
        return model
    
    def optimize_ranks(self, model: nn.Module, gradients: Dict[str, torch.Tensor]) -> nn.Module:
        """Optimize ranks of LoRA layers"""
        rank_allocation = self.rank_optimizer.optimize_rank_allocation(model, gradients)
        
        for name, target_rank in rank_allocation.items():
            module = model.get_submodule(name)
            adjusted_module = self.rank_optimizer.adjust_rank_dynamically(module, target_rank)
            
            # Update module in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, adjusted_module)
            else:
                setattr(model, child_name, adjusted_module)
        
        logger.info(f"Optimized ranks for {len(rank_allocation)} layers")
        return model
    
    def get_compression_metrics(self) -> Dict[str, Any]:
        """Get compression metrics"""
        total_compression = 0
        total_layers = len(self.quantization_stats)
        
        for stats in self.quantization_stats.values():
            total_compression += stats['compression_ratio']
        
        avg_compression = total_compression / total_layers if total_layers > 0 else 1.0
        
        return {
            'total_layers_quantized': total_layers,
            'average_compression_ratio': avg_compression,
            'quantization_stats': self.quantization_stats,
            'rank_optimization_stats': self.rank_optimizer.importance_scores
        }
    
    def calibrate_quantization(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """Calibrate quantization using representative data"""
        model.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(calibration_data[:self.quantization_config.calibration_samples]):
                # Forward pass to collect activation statistics
                _ = model(sample)
                
                if i % 10 == 0:
                    logger.info(f"Calibration progress: {i+1}/{len(calibration_data)}")
        
        logger.info("Quantization calibration complete")
    
    def save_quantization_config(self, filepath: str):
        """Save quantization configuration"""
        config = {
            'quantization_config': {
                'quant_type': self.quantization_config.quant_type.value,
                'use_double_quantization': self.quantization_config.use_double_quantization,
                'adaptive_bits': self.quantization_config.adaptive_bits,
                'compression_target': self.quantization_config.compression_target
            },
            'rank_config': {
                'adaptive_rank': self.rank_config.adaptive_rank,
                'min_rank': self.rank_config.min_rank,
                'max_rank': self.rank_config.max_rank
            },
            'quantization_stats': self.quantization_stats,
            'compression_metrics': self.get_compression_metrics()
        }
        
        torch.save(config, filepath)
        logger.info(f"Quantization config saved: {filepath}")
    
    def load_quantization_config(self, filepath: str):
        """Load quantization configuration"""
        config = torch.load(filepath, map_location='cpu')
        
        self.quantization_stats = config['quantization_stats']
        logger.info(f"Quantization config loaded: {filepath}")