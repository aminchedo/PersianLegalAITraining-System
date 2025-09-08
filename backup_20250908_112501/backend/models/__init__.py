# Models for Persian Legal AI System
# Note: Database models are imported conditionally to avoid import errors

# Training models
from .dora_trainer import DoRATrainer, DoRALayer
from .model_manager import ModelManager
from .qr_adaptor import QRAdaptor
from .verified_data_trainer import VerifiedDataTrainer

__all__ = [
    'DoRATrainer',
    'DoRALayer',
    'ModelManager',
    'QRAdaptor',
    'VerifiedDataTrainer'
]

# Try to import database models if they exist
try:
    from .user_model import User
    from .team_model import TeamMember
    from .model_training import ModelTraining
    from .system_metrics import SystemMetrics
    from .analytics_data import AnalyticsData
    from .legal_document import LegalDocument
    from .training_job import TrainingJob
    
    __all__.extend([
        'User',
        'TeamMember', 
        'ModelTraining',
        'SystemMetrics',
        'AnalyticsData',
        'LegalDocument',
        'TrainingJob'
    ])
except ImportError:
    # Database models not available, continue without them
    pass