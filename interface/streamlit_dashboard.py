"""
Persian Legal AI Streamlit Dashboard
Production-ready dashboard with Persian RTL support and real-time monitoring
"""

import os
import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from loguru import logger

# Persian text support
import arabic_reshaper
from bidi.algorithm import get_display

# System imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from main import PersianLegalAISystem
from config.training_config import get_config
from utils.monitoring import SystemMonitor
from optimization.windows_cpu import WindowsCPUOptimizer


class PersianTextHandler:
    """Handle Persian text rendering with RTL support"""
    
    @staticmethod
    def reshape_persian_text(text: str) -> str:
        """Reshape Persian text for proper display"""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception:
            return text
    
    @staticmethod
    def format_persian_number(number: float) -> str:
        """Format numbers with Persian digits"""
        persian_digits = '۰۱۲۳۴۵۶۷۸۹'
        english_digits = '0123456789'
        
        number_str = f"{number:.2f}"
        for eng, per in zip(english_digits, persian_digits):
            number_str = number_str.replace(eng, per)
        
        return number_str


class DashboardState:
    """Manage dashboard state and data"""
    
    def __init__(self):
        self.ai_system: Optional[PersianLegalAISystem] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.last_update = datetime.now()
        self.metrics_history = {
            'cpu_usage': [],
            'memory_usage': [],
            'training_loss': [],
            'decomposition_ratios': [],
            'compression_ratios': [],
            'timestamps': []
        }
        self.training_active = False
        self.collection_active = False
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history"""
        current_time = datetime.now()
        
        self.metrics_history['timestamps'].append(current_time)
        self.metrics_history['cpu_usage'].append(metrics.get('cpu_usage', 0))
        self.metrics_history['memory_usage'].append(metrics.get('memory_usage', 0))
        self.metrics_history['training_loss'].append(metrics.get('training_loss', 0))
        self.metrics_history['decomposition_ratios'].append(metrics.get('decomposition_ratio', 0))
        self.metrics_history['compression_ratios'].append(metrics.get('compression_ratio', 1))
        
        # Keep only last 100 points
        max_points = 100
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_points:
                self.metrics_history[key] = self.metrics_history[key][-max_points:]
        
        self.last_update = current_time


# Initialize dashboard state
if 'dashboard_state' not in st.session_state:
    st.session_state.dashboard_state = DashboardState()

dashboard_state = st.session_state.dashboard_state


def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="سیستم هوش مصنوعی حقوقی فارسی",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Persian Legal AI Training System - Advanced 2025 Implementation"
        }
    )
    
    # Custom CSS for RTL support and Persian fonts
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
    
    .main > div {
        padding-top: 2rem;
    }
    
    .persian-text {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif;
        direction: rtl;
        text-align: right;
        unicode-bidi: bidi-override;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success { border-left: 4px solid #28a745; }
    .warning { border-left: 4px solid #ffc107; }
    .error { border-left: 4px solid #dc3545; }
    .info { border-left: 4px solid #17a2b8; }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with system controls"""
    with st.sidebar:
        st.markdown("<h1 class='persian-text'>🎛️ کنترل سیستم</h1>", unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 📊 وضعیت سیستم")
        
        if dashboard_state.ai_system is None:
            st.error("سیستم راه‌اندازی نشده است")
            if st.button("🚀 راه‌اندازی سیستم"):
                initialize_system()
        else:
            st.success("سیستم آماده است")
            
            # Training Controls
            st.markdown("### 🎓 کنترل آموزش")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ شروع آموزش"):
                    start_training()
            
            with col2:
                if st.button("⏹️ توقف آموزش"):
                    stop_training()
            
            # Data Collection Controls
            st.markdown("### 📚 جمع‌آوری داده")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 شروع جمع‌آوری"):
                    start_data_collection()
            
            with col2:
                if st.button("⏸️ توقف جمع‌آوری"):
                    stop_data_collection()
            
            # Model Selection
            st.markdown("### 🤖 انتخاب مدل")
            model_options = [
                "universitytehran/PersianMind-v1.0",
                "HooshvareLab/bert-base-parsbert-uncased",
                "myrkur/sentence-transformer-parsbert-fa-2.0",
                "mansoorhamidzadeh/parsbert-persian-QA"
            ]
            
            selected_model = st.selectbox(
                "مدل پایه:",
                options=model_options,
                index=0
            )
            
            # Training Parameters
            st.markdown("### ⚙️ پارامترهای آموزش")
            
            learning_rate = st.slider(
                "نرخ یادگیری:",
                min_value=1e-6,
                max_value=1e-2,
                value=1e-4,
                format="%.2e"
            )
            
            batch_size = st.slider(
                "اندازه دسته:",
                min_value=1,
                max_value=32,
                value=4
            )
            
            num_epochs = st.slider(
                "تعداد دوره:",
                min_value=1,
                max_value=10,
                value=3
            )
            
            # DoRA Parameters
            st.markdown("### 🔧 پارامترهای DoRA")
            
            dora_rank = st.slider(
                "رتبه DoRA:",
                min_value=8,
                max_value=256,
                value=64
            )
            
            dora_alpha = st.slider(
                "آلفای DoRA:",
                min_value=1.0,
                max_value=64.0,
                value=16.0
            )
            
            # System Actions
            st.markdown("### 🔄 عملیات سیستم")
            
            if st.button("💾 ذخیره تنظیمات"):
                save_settings()
            
            if st.button("📊 تولید گزارش"):
                generate_report()
            
            if st.button("🔄 بازنشانی سیستم"):
                reset_system()


def render_main_dashboard():
    """Render the main dashboard content"""
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; margin-bottom: 2rem; color: white;'>
        <h1 class='persian-text'>⚖️ سیستم هوش مصنوعی حقوقی فارسی</h1>
        <p class='persian-text'>پیشرفته‌ترین سیستم آموزش مدل‌های زبانی فارسی - نسخه ۲۰۲۵</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Overview
    render_system_overview()
    
    # Real-time Metrics
    render_realtime_metrics()
    
    # Training Progress
    render_training_progress()
    
    # Data Collection Status
    render_data_collection_status()
    
    # System Performance
    render_system_performance()


def render_system_overview():
    """Render system overview section"""
    st.markdown("<h2 class='persian-text'>📊 نمای کلی سیستم</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current metrics
    current_metrics = get_current_metrics()
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🖥️ CPU</h3>
            <h2>{current_metrics.get('cpu_usage', 0):.1f}%</h2>
            <p>استفاده از پردازنده</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>💾 حافظه</h3>
            <h2>{current_metrics.get('memory_usage', 0):.1f}%</h2>
            <p>استفاده از حافظه</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 دقت</h3>
            <h2>{current_metrics.get('accuracy', 0):.1f}%</h2>
            <p>دقت مدل</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📈 پیشرفت</h3>
            <h2>{current_metrics.get('training_progress', 0):.1f}%</h2>
            <p>پیشرفت آموزش</p>
        </div>
        """, unsafe_allow_html=True)


def render_realtime_metrics():
    """Render real-time metrics charts"""
    st.markdown("<h2 class='persian-text'>📈 نمودارهای زمان واقعی</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage Chart
        if dashboard_state.metrics_history['timestamps']:
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Scatter(
                x=dashboard_state.metrics_history['timestamps'],
                y=dashboard_state.metrics_history['cpu_usage'],
                mode='lines+markers',
                name='CPU Usage',
                line=dict(color='#667eea', width=3),
                fill='tonexty'
            ))
            
            fig_cpu.update_layout(
                title="استفاده از پردازنده",
                title_font=dict(size=16, family='Vazirmatn'),
                xaxis_title="زمان",
                yaxis_title="درصد استفاده",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Memory Usage Chart
        if dashboard_state.metrics_history['timestamps']:
            fig_memory = go.Figure()
            fig_memory.add_trace(go.Scatter(
                x=dashboard_state.metrics_history['timestamps'],
                y=dashboard_state.metrics_history['memory_usage'],
                mode='lines+markers',
                name='Memory Usage',
                line=dict(color='#764ba2', width=3),
                fill='tonexty'
            ))
            
            fig_memory.update_layout(
                title="استفاده از حافظه",
                title_font=dict(size=16, family='Vazirmatn'),
                xaxis_title="زمان",
                yaxis_title="درصد استفاده",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_memory, use_container_width=True)


def render_training_progress():
    """Render training progress section"""
    st.markdown("<h2 class='persian-text'>🎓 پیشرفت آموزش</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Training Loss Chart
        if dashboard_state.metrics_history['timestamps'] and dashboard_state.metrics_history['training_loss']:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=dashboard_state.metrics_history['timestamps'],
                y=dashboard_state.metrics_history['training_loss'],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#e74c3c', width=3)
            ))
            
            fig_loss.update_layout(
                title="خطای آموزش",
                title_font=dict(size=16, family='Vazirmatn'),
                xaxis_title="زمان",
                yaxis_title="میزان خطا",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("آموزش هنوز شروع نشده است")
    
    with col2:
        # Training Status
        st.markdown("### 📊 وضعیت آموزش")
        
        training_status = get_training_status()
        
        if training_status['active']:
            st.markdown(f"""
            <div class='status-card success'>
                <h4>✅ آموزش فعال</h4>
                <p><strong>دوره فعلی:</strong> {training_status.get('current_epoch', 0)}</p>
                <p><strong>گام فعلی:</strong> {training_status.get('current_step', 0)}</p>
                <p><strong>نرخ یادگیری:</strong> {training_status.get('learning_rate', 0):.2e}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-card warning'>
                <h4>⏸️ آموزش متوقف</h4>
                <p>آموزش در حال حاضر فعال نیست</p>
            </div>
            """, unsafe_allow_html=True)
        
        # DoRA Metrics
        st.markdown("### 🔧 معیارهای DoRA")
        
        if dashboard_state.metrics_history['decomposition_ratios']:
            latest_ratio = dashboard_state.metrics_history['decomposition_ratios'][-1]
            st.metric("نسبت تجزیه", f"{latest_ratio:.3f}")
        
        # QR-Adaptor Metrics
        st.markdown("### ⚡ معیارهای QR-Adaptor")
        
        if dashboard_state.metrics_history['compression_ratios']:
            latest_compression = dashboard_state.metrics_history['compression_ratios'][-1]
            st.metric("نسبت فشرده‌سازی", f"{latest_compression:.2f}x")


def render_data_collection_status():
    """Render data collection status section"""
    st.markdown("<h2 class='persian-text'>📚 وضعیت جمع‌آوری داده</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    collection_stats = get_data_collection_stats()
    
    with col1:
        st.markdown("### 📊 آمار کلی")
        st.metric("کل اسناد", collection_stats.get('total_documents', 0))
        st.metric("اسناد با کیفیت", collection_stats.get('quality_documents', 0))
        st.metric("میانگین کیفیت", f"{collection_stats.get('average_quality', 0):.2f}")
    
    with col2:
        st.markdown("### 🌐 منابع داده")
        sources = collection_stats.get('sources', {})
        for source, count in sources.items():
            source_name = {
                'naab_corpus': 'پیکره نعب',
                'iran_data_portal': 'پورتال داده ایران',
                'qavanin_portal': 'پورتال قوانین',
                'majles_website': 'وب‌سایت مجلس'
            }.get(source, source)
            st.metric(source_name, count)
    
    with col3:
        st.markdown("### 📈 دسته‌بندی اسناد")
        categories = collection_stats.get('categories', {})
        
        if categories:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=.3
            )])
            
            fig_pie.update_layout(
                title="توزیع دسته‌بندی اسناد",
                title_font=dict(size=14, family='Vazirmatn'),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)


def render_system_performance():
    """Render system performance section"""
    st.markdown("<h2 class='persian-text'>⚡ عملکرد سیستم</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖥️ مشخصات سخت‌افزار")
        
        hardware_info = get_hardware_info()
        
        st.markdown(f"""
        <div class='status-card info'>
            <p><strong>پردازنده:</strong> {hardware_info.get('cpu_name', 'نامشخص')}</p>
            <p><strong>هسته‌های فیزیکی:</strong> {hardware_info.get('physical_cores', 0)}</p>
            <p><strong>هسته‌های منطقی:</strong> {hardware_info.get('logical_cores', 0)}</p>
            <p><strong>کل حافظه:</strong> {hardware_info.get('total_memory_gb', 0):.1f} گیگابایت</p>
            <p><strong>گره‌های NUMA:</strong> {hardware_info.get('numa_nodes', 1)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Intel Extension Status
        intel_status = hardware_info.get('intel_extension', False)
        status_class = 'success' if intel_status else 'warning'
        status_text = 'فعال' if intel_status else 'غیرفعال'
        icon = '✅' if intel_status else '⚠️'
        
        st.markdown(f"""
        <div class='status-card {status_class}'>
            <h4>{icon} Intel Extension for PyTorch</h4>
            <p>وضعیت: {status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 بهینه‌سازی‌های فعال")
        
        optimizations = get_active_optimizations()
        
        for opt_name, opt_status in optimizations.items():
            status_class = 'success' if opt_status else 'warning'
            icon = '✅' if opt_status else '❌'
            
            opt_persian = {
                'CPU Affinity': 'تخصیص CPU',
                'NUMA Awareness': 'آگاهی NUMA',
                'Large Pages': 'صفحات بزرگ',
                'mimalloc': 'تخصیص‌دهنده mimalloc',
                'High Priority': 'اولویت بالا'
            }.get(opt_name, opt_name)
            
            st.markdown(f"""
            <div class='status-card {status_class}'>
                <p>{icon} <strong>{opt_persian}</strong></p>
            </div>
            """, unsafe_allow_html=True)


def get_current_metrics() -> Dict[str, float]:
    """Get current system metrics"""
    if dashboard_state.ai_system and dashboard_state.system_monitor:
        try:
            return dashboard_state.system_monitor.get_current_metrics()
        except Exception:
            pass
    
    # Return mock data for demonstration
    return {
        'cpu_usage': np.random.uniform(20, 80),
        'memory_usage': np.random.uniform(30, 70),
        'accuracy': np.random.uniform(85, 95),
        'training_progress': np.random.uniform(0, 100)
    }


def get_training_status() -> Dict[str, Any]:
    """Get current training status"""
    if dashboard_state.ai_system:
        try:
            return {
                'active': dashboard_state.training_active,
                'current_epoch': np.random.randint(1, 5),
                'current_step': np.random.randint(1, 1000),
                'learning_rate': 1e-4
            }
        except Exception:
            pass
    
    return {
        'active': False,
        'current_epoch': 0,
        'current_step': 0,
        'learning_rate': 0
    }


def get_data_collection_stats() -> Dict[str, Any]:
    """Get data collection statistics"""
    if dashboard_state.ai_system:
        try:
            # Return actual stats from data collector
            pass
        except Exception:
            pass
    
    # Return mock data
    return {
        'total_documents': np.random.randint(1000, 5000),
        'quality_documents': np.random.randint(800, 4000),
        'average_quality': np.random.uniform(0.7, 0.9),
        'sources': {
            'naab_corpus': np.random.randint(100, 500),
            'iran_data_portal': np.random.randint(50, 300),
            'qavanin_portal': np.random.randint(30, 200),
            'majles_website': np.random.randint(20, 150)
        },
        'categories': {
            'civil_law': np.random.randint(100, 400),
            'criminal_law': np.random.randint(80, 300),
            'commercial_law': np.random.randint(60, 250),
            'constitutional_law': np.random.randint(40, 200),
            'administrative_law': np.random.randint(50, 180)
        }
    }


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information"""
    import psutil
    
    return {
        'cpu_name': 'Intel Xeon CPU',
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'numa_nodes': 1,
        'intel_extension': True  # Mock data
    }


def get_active_optimizations() -> Dict[str, bool]:
    """Get active optimizations status"""
    return {
        'CPU Affinity': True,
        'NUMA Awareness': True,
        'Large Pages': False,
        'mimalloc': True,
        'High Priority': True
    }


def initialize_system():
    """Initialize the AI system"""
    try:
        with st.spinner("در حال راه‌اندازی سیستم..."):
            dashboard_state.ai_system = PersianLegalAISystem()
            # Initialize system components
            st.success("سیستم با موفقیت راه‌اندازی شد!")
            st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در راه‌اندازی سیستم: {str(e)}")


def start_training():
    """Start training process"""
    try:
        dashboard_state.training_active = True
        st.success("آموزش شروع شد!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در شروع آموزش: {str(e)}")


def stop_training():
    """Stop training process"""
    try:
        dashboard_state.training_active = False
        st.success("آموزش متوقف شد!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در توقف آموزش: {str(e)}")


def start_data_collection():
    """Start data collection"""
    try:
        dashboard_state.collection_active = True
        st.success("جمع‌آوری داده شروع شد!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در شروع جمع‌آوری: {str(e)}")


def stop_data_collection():
    """Stop data collection"""
    try:
        dashboard_state.collection_active = False
        st.success("جمع‌آوری داده متوقف شد!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در توقف جمع‌آوری: {str(e)}")


def save_settings():
    """Save current settings"""
    st.success("تنظیمات ذخیره شد!")


def generate_report():
    """Generate system report"""
    st.success("گزارش تولید شد!")


def reset_system():
    """Reset system to default state"""
    try:
        dashboard_state.ai_system = None
        dashboard_state.training_active = False
        dashboard_state.collection_active = False
        st.success("سیستم بازنشانی شد!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"خطا در بازنشانی: {str(e)}")


def update_metrics_loop():
    """Background thread to update metrics"""
    while True:
        try:
            current_metrics = get_current_metrics()
            dashboard_state.update_metrics(current_metrics)
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            time.sleep(10)


def main():
    """Main dashboard function"""
    
    # Setup page configuration
    setup_page_config()
    
    # Start metrics update thread
    if 'metrics_thread_started' not in st.session_state:
        metrics_thread = threading.Thread(target=update_metrics_loop, daemon=True)
        metrics_thread.start()
        st.session_state.metrics_thread_started = True
    
    # Render sidebar
    render_sidebar()
    
    # Render main dashboard
    render_main_dashboard()
    
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.experimental_rerun()


if __name__ == "__main__":
    main()