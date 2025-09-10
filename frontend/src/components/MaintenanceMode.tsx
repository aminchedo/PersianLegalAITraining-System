import React from 'react';
import { AlertTriangle, Clock, Database, Cpu, Network } from 'lucide-react';

interface MaintenanceModeProps {
  title?: string;
  message?: string;
  showDetails?: boolean;
}

const MaintenanceMode: React.FC<MaintenanceModeProps> = ({
  title = "Backend API Under Maintenance",
  message = "The Persian Legal AI backend is currently undergoing dependency installation and configuration.",
  showDetails = true
}) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="mx-auto w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mb-4">
            <AlertTriangle className="w-8 h-8 text-orange-600" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">{title}</h1>
          <p className="text-gray-600">{message}</p>
        </div>

        {/* Status Badge */}
        <div className="flex justify-center mb-6">
          <span className="inline-flex items-center px-4 py-2 rounded-full bg-orange-100 text-orange-800 text-sm font-medium">
            <Clock className="w-4 h-4 mr-2" />
            Maintenance in Progress
          </span>
        </div>

        {/* Persian Text */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6 text-right" dir="rtl">
          <p className="text-gray-700 font-medium">
            سیستم آموزش هوش مصنوعی حقوقی فارسی
          </p>
          <p className="text-gray-600 text-sm mt-1">
            سرویس بک‌اند در حال تعمیر و نگهداری است
          </p>
        </div>

        {showDetails && (
          <>
            {/* System Status */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <div className="flex items-center">
                  <Database className="w-5 h-5 text-green-600 mr-2" />
                  <span className="text-sm font-medium text-green-800">Database</span>
                </div>
                <p className="text-xs text-green-600 mt-1">✅ Operational</p>
              </div>
              
              <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                <div className="flex items-center">
                  <Cpu className="w-5 h-5 text-orange-600 mr-2" />
                  <span className="text-sm font-medium text-orange-800">AI Models</span>
                </div>
                <p className="text-xs text-orange-600 mt-1">🔧 Installing</p>
              </div>
              
              <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                <div className="flex items-center">
                  <Network className="w-5 h-5 text-red-600 mr-2" />
                  <span className="text-sm font-medium text-red-800">API</span>
                </div>
                <p className="text-xs text-red-600 mt-1">❌ Unavailable</p>
              </div>
            </div>

            {/* Details */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <h3 className="font-semibold text-blue-900 mb-2">What's Happening?</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Installing 45+ missing Python dependencies</li>
                <li>• Setting up virtual environment</li>
                <li>• Configuring AI/ML libraries (PyTorch, Transformers)</li>
                <li>• Installing Persian NLP tools (Hazm)</li>
              </ul>
            </div>

            {/* Timeline */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-semibold text-gray-900 mb-2">Recovery Timeline</h3>
              <div className="space-y-2 text-sm text-gray-700">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-orange-400 rounded-full mr-3"></div>
                  <span>Phase 1: Dependency Installation (30-60 min)</span>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-gray-300 rounded-full mr-3"></div>
                  <span>Phase 2: Configuration Setup (15-30 min)</span>
                </div>
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-gray-300 rounded-full mr-3"></div>
                  <span>Phase 3: System Verification (30-45 min)</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                <strong>Estimated Total Time:</strong> 3-6 hours
              </p>
            </div>
          </>
        )}

        {/* Footer */}
        <div className="text-center text-sm text-gray-500">
          <p>
            For technical details, see the{' '}
            <a 
              href="https://github.com/aminchedo/PersianLegalAITraining-System/blob/main/PERSIAN_LEGAL_AI_COMPREHENSIVE_AUDIT_REPORT.md" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              comprehensive audit report
            </a>
          </p>
          <p className="mt-2">Last updated: {new Date().toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
};

export default MaintenanceMode;