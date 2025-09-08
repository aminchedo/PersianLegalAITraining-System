"""
Enhanced API Endpoints (Preserves existing routes)
================================================
Adds enhanced functionality without breaking existing API
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

from backend.services.enhanced_model_service import enhanced_model_service

logger = logging.getLogger(__name__)

# Create enhanced router with different prefix
enhanced_router = APIRouter(prefix="/api/enhanced", tags=["Enhanced Features"])

class EnhancedRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Persian text for enhanced processing")

class EnhancedResponse(BaseModel):
    classification: str
    confidence: float
    all_scores: Dict[str, float]
    processing_time: Optional[float] = None
    method: str
    model_source: str

@enhanced_router.get("/status")
async def get_enhanced_status():
    """Get enhanced service status (doesn't interfere with existing /status)"""
    try:
        status_data = enhanced_model_service.get_enhanced_status()
        return status_data
    except Exception as e:
        logger.error(f"Enhanced status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced status error: {str(e)}"
        )

@enhanced_router.post("/classify", response_model=EnhancedResponse)
async def enhanced_classify(request: EnhancedRequest):
    """Enhanced classification (coexists with existing /classify)"""
    try:
        result = enhanced_model_service.classify_enhanced(request.text)
        
        return EnhancedResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            all_scores=result.get("all_scores", {}),
            processing_time=result.get("processing_time"),
            method=result.get("method", "unknown"),
            model_source=result.get("model_source", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Enhanced classification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced classification failed: {str(e)}"
        )

@enhanced_router.post("/compare")
async def compare_classifications(request: EnhancedRequest):
    """Compare enhanced vs existing classification methods"""
    try:
        # Get enhanced result
        enhanced_result = enhanced_model_service.classify_enhanced(request.text)
        
        # Try to get existing result if service available
        existing_result = None
        if enhanced_model_service.existing_service:
            try:
                if hasattr(enhanced_model_service.existing_service, 'classify_document'):
                    existing_result = enhanced_model_service.existing_service.classify_document(request.text)
            except Exception as e:
                logger.warning(f"Existing service comparison failed: {e}")
        
        return {
            "enhanced": enhanced_result,
            "existing": existing_result,
            "text_length": len(request.text)
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )