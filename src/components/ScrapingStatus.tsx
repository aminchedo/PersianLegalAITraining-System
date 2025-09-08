import React, { useState } from 'react';
import { Play, Square, AlertCircle, CheckCircle, Clock, Globe } from 'lucide-react';
import { useScrapingStatus, useStartScraping, useStopScraping } from '../hooks/useDocuments';
import { IRANIAN_LEGAL_SOURCES } from '../types';

export const ScrapingStatus: React.FC = () => {
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  
  const { data: status, isLoading } = useScrapingStatus();
  const startScrapingMutation = useStartScraping();
  const stopScrapingMutation = useStopScraping();

  const handleSourceToggle = (source: string) => {
    setSelectedSources(prev => 
      prev.includes(source) 
        ? prev.filter(s => s !== source)
        : [...prev, source]
    );
  };

  const handleStartScraping = () => {
    if (selectedSources.length > 0) {
      startScrapingMutation.mutate(selectedSources);
    }
  };

  const handleStopScraping = () => {
    stopScrapingMutation.mutate();
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-8 bg-gray-200 rounded w-full"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">وضعیت جمع‌آوری اسناد</h2>
        <div className="flex items-center gap-2">
          {status?.is_running ? (
            <div className="flex items-center gap-2 text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium">در حال اجرا</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-gray-500">
              <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
              <span className="text-sm font-medium">متوقف</span>
            </div>
          )}
        </div>
      </div>

      {/* Current Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Globe className="w-5 h-5 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">سایت فعلی</span>
          </div>
          <p className="text-lg font-semibold text-blue-900">
            {status?.current_site || 'هیچکدام'}
          </p>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-green-800">اسناد جمع‌آوری شده</span>
          </div>
          <p className="text-lg font-semibold text-green-900">
            {status?.documents_scraped || 0}
          </p>
        </div>

        <div className="bg-yellow-50 p-4 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-yellow-600" />
            <span className="text-sm font-medium text-yellow-800">زمان شروع</span>
          </div>
          <p className="text-sm font-medium text-yellow-900">
            {status?.started_at 
              ? new Date(status.started_at).toLocaleString('fa-IR')
              : 'شروع نشده'
            }
          </p>
        </div>
      </div>

      {/* Source Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-900 mb-3">انتخاب منابع</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {IRANIAN_LEGAL_SOURCES.map((source) => (
            <label key={source} className="flex items-center">
              <input
                type="checkbox"
                checked={selectedSources.includes(source)}
                onChange={() => handleSourceToggle(source)}
                disabled={status?.is_running}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 ml-2"
              />
              <span className="text-sm text-gray-700">{source}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handleStartScraping}
          disabled={status?.is_running || selectedSources.length === 0 || startScrapingMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Play className="w-4 h-4" />
          شروع جمع‌آوری
        </button>

        <button
          onClick={handleStopScraping}
          disabled={!status?.is_running || stopScrapingMutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Square className="w-4 h-4" />
          توقف جمع‌آوری
        </button>
      </div>

      {/* Errors */}
      {status?.errors && status.errors.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <h4 className="font-medium text-red-800">خطاها</h4>
          </div>
          <ul className="text-sm text-red-700 space-y-1">
            {status.errors.map((error, index) => (
              <li key={index} className="list-disc list-inside">
                {error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Progress Bar */}
      {status?.is_running && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>پیشرفت جمع‌آوری</span>
            <span>{status.documents_scraped} سند</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full animate-pulse" 
              style={{ width: '100%' }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScrapingStatus;