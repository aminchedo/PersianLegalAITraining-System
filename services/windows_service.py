"""
Windows Service Configuration for Persian Legal AI System
Enables 24/7 operation as a Windows service with automatic recovery
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import signal

# Windows service imports
try:
    import win32serviceutil
    import win32service
    import win32event
    import win32api
    import servicemanager
    import win32con
    import win32evtlogutil
    WINDOWS_SERVICE_AVAILABLE = True
except ImportError:
    WINDOWS_SERVICE_AVAILABLE = False

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from main import PersianLegalAISystem
from config.training_config import get_config
from loguru import logger


class PersianLegalAIService(win32serviceutil.ServiceFramework):
    """Windows Service for Persian Legal AI System"""
    
    _svc_name_ = "PersianLegalAI"
    _svc_display_name_ = "Persian Legal AI Training System"
    _svc_description_ = "Advanced Persian Legal AI Training System with DoRA and QR-Adaptor - 2025 Edition"
    _svc_deps_ = None  # Dependencies (none required)
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        
        # Event to signal service stop
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        
        # Service state
        self.is_running = False
        self.ai_system: Optional[PersianLegalAISystem] = None
        self.main_thread: Optional[threading.Thread] = None
        
        # Setup service logging
        self._setup_service_logging()
        
        logger.info("Persian Legal AI Service initialized")
    
    def _setup_service_logging(self) -> None:
        """Setup logging for Windows service"""
        
        # Create logs directory
        log_dir = Path("C:/ProgramData/PersianLegalAI/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove default loguru handler
        logger.remove()
        
        # Add file logging for service
        logger.add(
            log_dir / "service_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            compression="zip"
        )
        
        # Add Windows Event Log
        try:
            win32evtlogutil.AddSourceToRegistry(
                appName=self._svc_name_,
                msgDLL=win32evtlogutil.GetEventLogMessageExe(),
                eventLogType="Application"
            )
            
            def windows_event_sink(message):
                try:
                    event_type = win32evtlog.EVENTLOG_INFORMATION_TYPE
                    if message.record["level"].name == "ERROR":
                        event_type = win32evtlog.EVENTLOG_ERROR_TYPE
                    elif message.record["level"].name == "WARNING":
                        event_type = win32evtlog.EVENTLOG_WARNING_TYPE
                    
                    win32evtlogutil.ReportEvent(
                        self._svc_name_,
                        1000,
                        eventCategory=0,
                        eventType=event_type,
                        strings=[message.record["message"]]
                    )
                except Exception:
                    pass
            
            logger.add(windows_event_sink, level="WARNING")
            
        except Exception as e:
            # Continue without Windows Event Log if it fails
            pass
    
    def SvcStop(self):
        """Service stop handler"""
        logger.info("Persian Legal AI Service stop requested")
        
        # Report service stop pending
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
        # Signal the main thread to stop
        self.is_running = False
        win32event.SetEvent(self.hWaitStop)
        
        # Stop the AI system gracefully
        if self.ai_system:
            try:
                asyncio.run(self.ai_system.shutdown())
            except Exception as e:
                logger.error(f"Error during AI system shutdown: {e}")
        
        logger.info("Persian Legal AI Service stopped")
    
    def SvcDoRun(self):
        """Main service execution"""
        logger.info("Persian Legal AI Service starting...")
        
        # Report service start
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        try:
            # Set service status to running
            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            self.is_running = True
            
            # Start the main service thread
            self.main_thread = threading.Thread(target=self._run_main_loop, daemon=False)
            self.main_thread.start()
            
            # Wait for stop event
            win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
            
            # Wait for main thread to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=30)
            
        except Exception as e:
            logger.error(f"Service execution error: {e}")
            servicemanager.LogErrorMsg(f"Persian Legal AI Service error: {str(e)}")
        
        # Report service stopped
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, '')
        )
    
    def _run_main_loop(self):
        """Main service loop"""
        
        retry_count = 0
        max_retries = 5
        
        while self.is_running and retry_count < max_retries:
            try:
                logger.info("Initializing Persian Legal AI System...")
                
                # Initialize the AI system
                self.ai_system = PersianLegalAISystem()
                
                # Run the system
                asyncio.run(self._run_ai_system())
                
                # If we get here without exception, reset retry count
                retry_count = 0
                
            except Exception as e:
                retry_count += 1
                logger.error(f"AI System error (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count < max_retries:
                    logger.info(f"Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    logger.error("Maximum retries reached. Service will stop.")
                    break
        
        logger.info("Main service loop ended")
    
    async def _run_ai_system(self):
        """Run the AI system asynchronously"""
        
        try:
            # Initialize system
            if not await self.ai_system.initialize_system():
                raise RuntimeError("Failed to initialize AI system")
            
            logger.info("AI System initialized successfully")
            
            # Start training if configured
            config = get_config()
            if hasattr(config, 'auto_start_training') and config.auto_start_training:
                logger.info("Auto-starting training...")
                await self.ai_system.start_training()
            
            # Keep the service running
            while self.is_running:
                await asyncio.sleep(10)
                
                # Periodic health check
                if not self.ai_system.is_initialized:
                    logger.warning("AI system lost initialization - restarting...")
                    break
            
        except Exception as e:
            logger.error(f"AI system runtime error: {e}")
            raise
        finally:
            if self.ai_system:
                await self.ai_system.shutdown()


class WindowsServiceManager:
    """Manage Windows service operations"""
    
    def __init__(self):
        self.service_name = "PersianLegalAI"
        self.available = WINDOWS_SERVICE_AVAILABLE
        
        if not self.available:
            logger.warning("Windows service functionality not available")
    
    def install_service(self) -> bool:
        """Install the Windows service"""
        
        if not self.available:
            logger.error("Windows service not available")
            return False
        
        try:
            # Get the path to the current script
            service_script = Path(__file__).absolute()
            
            # Install service
            win32serviceutil.InstallService(
                PersianLegalAIService._svc_reg_class_,
                PersianLegalAIService._svc_name_,
                PersianLegalAIService._svc_display_name_,
                description=PersianLegalAIService._svc_description_,
                startType=win32service.SERVICE_AUTO_START,
                errorControl=win32service.SERVICE_ERROR_NORMAL,
                exeName=str(service_script)
            )
            
            logger.success(f"Service '{self.service_name}' installed successfully")
            
            # Set service recovery options
            self._configure_service_recovery()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return False
    
    def _configure_service_recovery(self):
        """Configure service recovery options"""
        
        try:
            # Open service manager
            scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
            
            # Open the service
            service = win32service.OpenService(scm, self.service_name, win32service.SERVICE_ALL_ACCESS)
            
            # Configure failure actions
            # Restart after 1 minute, restart after 5 minutes, restart after 10 minutes
            failure_actions = [
                (win32service.SC_ACTION_RESTART, 60000),  # 1 minute
                (win32service.SC_ACTION_RESTART, 300000),  # 5 minutes
                (win32service.SC_ACTION_RESTART, 600000),  # 10 minutes
            ]
            
            # Set failure actions
            win32service.ChangeServiceConfig2(
                service,
                win32service.SERVICE_CONFIG_FAILURE_ACTIONS,
                {
                    'ResetPeriod': 86400,  # Reset after 24 hours
                    'RebootMsg': '',
                    'Command': '',
                    'Actions': failure_actions
                }
            )
            
            logger.info("Service recovery options configured")
            
            # Close handles
            win32service.CloseServiceHandle(service)
            win32service.CloseServiceHandle(scm)
            
        except Exception as e:
            logger.warning(f"Could not configure service recovery: {e}")
    
    def uninstall_service(self) -> bool:
        """Uninstall the Windows service"""
        
        if not self.available:
            logger.error("Windows service not available")
            return False
        
        try:
            # Stop service first if running
            self.stop_service()
            
            # Uninstall service
            win32serviceutil.RemoveService(self.service_name)
            
            logger.success(f"Service '{self.service_name}' uninstalled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return False
    
    def start_service(self) -> bool:
        """Start the Windows service"""
        
        if not self.available:
            logger.error("Windows service not available")
            return False
        
        try:
            win32serviceutil.StartService(self.service_name)
            logger.success(f"Service '{self.service_name}' started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the Windows service"""
        
        if not self.available:
            logger.error("Windows service not available")
            return False
        
        try:
            win32serviceutil.StopService(self.service_name)
            logger.success(f"Service '{self.service_name}' stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            return False
    
    def restart_service(self) -> bool:
        """Restart the Windows service"""
        
        if not self.available:
            logger.error("Windows service not available")
            return False
        
        try:
            self.stop_service()
            time.sleep(5)  # Wait a bit
            return self.start_service()
            
        except Exception as e:
            logger.error(f"Failed to restart service: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status information"""
        
        if not self.available:
            return {'available': False, 'status': 'unavailable'}
        
        try:
            # Open service manager
            scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
            
            # Open service
            service = win32service.OpenService(scm, self.service_name, win32service.SERVICE_QUERY_STATUS)
            
            # Query status
            status = win32service.QueryServiceStatus(service)
            
            # Close handles
            win32service.CloseServiceHandle(service)
            win32service.CloseServiceHandle(scm)
            
            # Map status codes to readable names
            status_map = {
                win32service.SERVICE_STOPPED: 'stopped',
                win32service.SERVICE_START_PENDING: 'starting',
                win32service.SERVICE_STOP_PENDING: 'stopping',
                win32service.SERVICE_RUNNING: 'running',
                win32service.SERVICE_CONTINUE_PENDING: 'resuming',
                win32service.SERVICE_PAUSE_PENDING: 'pausing',
                win32service.SERVICE_PAUSED: 'paused'
            }
            
            return {
                'available': True,
                'status': status_map.get(status[1], 'unknown'),
                'status_code': status[1],
                'controls_accepted': status[2],
                'exit_code': status[3],
                'service_exit_code': status[4],
                'check_point': status[5],
                'wait_hint': status[6]
            }
            
        except Exception as e:
            return {
                'available': True,
                'status': 'error',
                'error': str(e)
            }
    
    def run(self) -> int:
        """Run service based on command line arguments"""
        
        if not self.available:
            print("Windows service functionality not available")
            return 1
        
        if len(sys.argv) == 1:
            # Run as service
            servicemanager.Initialize()
            servicemanager.PrepareToHostSingle(PersianLegalAIService)
            servicemanager.StartServiceCtrlDispatcher()
            return 0
        
        # Handle command line arguments
        command = sys.argv[1].lower()
        
        if command == 'install':
            return 0 if self.install_service() else 1
        elif command == 'remove' or command == 'uninstall':
            return 0 if self.uninstall_service() else 1
        elif command == 'start':
            return 0 if self.start_service() else 1
        elif command == 'stop':
            return 0 if self.stop_service() else 1
        elif command == 'restart':
            return 0 if self.restart_service() else 1
        elif command == 'status':
            status = self.get_service_status()
            print(f"Service Status: {status}")
            return 0
        else:
            print(f"Usage: {sys.argv[0]} [install|remove|start|stop|restart|status]")
            return 1


def create_service_installer_script():
    """Create a batch script for easy service installation"""
    
    script_content = f"""@echo off
REM Persian Legal AI Service Installer
REM Run as Administrator

echo Persian Legal AI Service Management
echo ===================================

:menu
echo.
echo 1. Install Service
echo 2. Uninstall Service
echo 3. Start Service
echo 4. Stop Service
echo 5. Restart Service
echo 6. Check Status
echo 7. Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto uninstall
if "%choice%"=="3" goto start
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto restart
if "%choice%"=="6" goto status
if "%choice%"=="7" goto exit

echo Invalid choice. Please try again.
goto menu

:install
echo Installing Persian Legal AI Service...
python "{Path(__file__).absolute()}" install
pause
goto menu

:uninstall
echo Uninstalling Persian Legal AI Service...
python "{Path(__file__).absolute()}" remove
pause
goto menu

:start
echo Starting Persian Legal AI Service...
python "{Path(__file__).absolute()}" start
pause
goto menu

:stop
echo Stopping Persian Legal AI Service...
python "{Path(__file__).absolute()}" stop
pause
goto menu

:restart
echo Restarting Persian Legal AI Service...
python "{Path(__file__).absolute()}" restart
pause
goto menu

:status
echo Checking Persian Legal AI Service Status...
python "{Path(__file__).absolute()}" status
pause
goto menu

:exit
echo Goodbye!
pause
"""
    
    # Write the batch script
    script_path = Path(__file__).parent / "service_manager.bat"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Service installer script created: {script_path}")


if __name__ == "__main__":
    # Create service installer script
    create_service_installer_script()
    
    # Run service manager
    service_manager = WindowsServiceManager()
    exit_code = service_manager.run()
    sys.exit(exit_code)