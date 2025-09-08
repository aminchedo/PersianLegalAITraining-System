import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Send, FileText, BarChart3, Clock, Target } from 'lucide-react';

interface ClassificationResult {
  category: string;
  confidence: number;
  processing_time_ms: number;
  model_info: {
    model_name: string;
    version: string;
  };
  detailed_scores: Record<string, number>;
}

const ClassificationPage: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState<ClassificationResult | null>(null);

  // Classification mutation
  const classifyMutation = useMutation({
    mutationFn: async (text: string) => {
      const response = await fetch('/api/classification/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      return response.json();
    },
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleClassify = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      classifyMutation.mutate(inputText);
    }
  };

  const categories = [
    { key: 'حقوق مدنی', label: 'حقوق مدنی', color: 'bg-blue-500' },
    { key: 'حقوق کیفری', label: 'حقوق کیفری', color: 'bg-red-500' },
    { key: 'حقوق اداری', label: 'حقوق اداری', color: 'bg-green-500' },
    { key: 'حقوق تجاری', label: 'حقوق تجاری', color: 'bg-yellow-500' },
    { key: 'حقوق اساسی', label: 'حقوق اساسی', color: 'bg-purple-500' },
    { key: 'رأی قضایی', label: 'رأی قضایی', color: 'bg-indigo-500' },
    { key: 'بخشنامه', label: 'بخشنامه', color: 'bg-pink-500' },
  ];

  const sampleTexts = [
    {
      title: 'نمونه حقوق مدنی',
      text: 'طبق ماده ۱۰ قانون مدنی، هر شخص از زمان تولد تا زمان مرگ دارای شخصیت حقوقی است و در حدود قانون از حقوق مدنی برخوردار می‌باشد.',
      category: 'حقوق مدنی'
    },
    {
      title: 'نمونه حقوق کیفری',
      text: 'مرتکب جرم قتل عمد به قصاص محکوم می‌شود مگر اینکه اولیای دم عفو کنند یا دیه را بپذیرند.',
      category: 'حقوق کیفری'
    },
    {
      title: 'نمونه رأی قضایی',
      text: 'دیوان عدالت اداری با بررسی پرونده و مستندات ارائه شده و با عنایت به اینکه تصمیم مرجع اداری مخالف قوانین بوده، تصمیم مذکور را باطل اعلام می‌کند.',
      category: 'رأی قضایی'
    }
  ];

  const getCategoryColor = (category: string) => {
    const cat = categories.find(c => c.key === category);
    return cat ? cat.color : 'bg-gray-500';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          طبقه‌بندی متون حقوقی
        </h1>
        <p className="text-gray-600">
          طبقه‌بندی خودکار متون حقوقی فارسی با استفاده از مدل BERT بهینه‌شده
        </p>
      </div>

      {/* Input Section */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          متن مورد نظر را وارد کنید
        </h2>
        
        <form onSubmit={handleClassify} className="space-y-4">
          <div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="متن حقوقی فارسی را اینجا وارد کنید..."
              rows={8}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            />
          </div>
          
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">
              تعداد کاراکتر: {inputText.length}
            </span>
            <button
              type="submit"
              disabled={!inputText.trim() || classifyMutation.isPending}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg flex items-center"
            >
              {classifyMutation.isPending ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white ml-2"></div>
                  در حال تحلیل...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4 ml-2" />
                  طبقه‌بندی کن
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Sample Texts */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          متون نمونه
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {sampleTexts.map((sample, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition-colors"
              onClick={() => setInputText(sample.text)}
            >
              <h3 className="font-medium text-gray-900 mb-2">{sample.title}</h3>
              <p className="text-sm text-gray-600 line-clamp-3 mb-3">
                {sample.text}
              </p>
              <span className={`inline-block px-2 py-1 text-xs text-white rounded-full ${getCategoryColor(sample.category)}`}>
                {sample.category}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            نتایج طبقه‌بندی
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Main Result */}
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">دسته‌بندی پیش‌بینی شده</h3>
                  <Target className="h-6 w-6 text-blue-600" />
                </div>
                
                <div className="text-center">
                  <div className={`inline-block px-6 py-3 text-white rounded-full text-lg font-bold ${getCategoryColor(result.category)}`}>
                    {result.category}
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mt-4">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-gray-600">اطمینان</p>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4 text-center">
                  <Clock className="h-6 w-6 text-gray-600 mx-auto mb-2" />
                  <p className="text-lg font-bold text-gray-900">
                    {result.processing_time_ms}ms
                  </p>
                  <p className="text-sm text-gray-600">زمان پردازش</p>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-4 text-center">
                  <FileText className="h-6 w-6 text-gray-600 mx-auto mb-2" />
                  <p className="text-lg font-bold text-gray-900">
                    {result.model_info.model_name.split('/')[1] || 'BERT'}
                  </p>
                  <p className="text-sm text-gray-600">مدل استفاده شده</p>
                </div>
              </div>
            </div>

            {/* Detailed Scores */}
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">امتیازات تفصیلی</h3>
                <BarChart3 className="h-6 w-6 text-gray-600" />
              </div>
              
              <div className="space-y-3">
                {Object.entries(result.detailed_scores)
                  .sort(([,a], [,b]) => b - a)
                  .map(([category, score]) => (
                    <div key={category} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">
                          {category}
                        </span>
                        <span className="text-sm text-gray-600">
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${
                            category === result.category ? getCategoryColor(category) : 'bg-gray-400'
                          }`}
                          style={{ width: `${score * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))
                }
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {classifyMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="mr-3">
              <h3 className="text-sm font-medium text-red-800">
                خطا در طبقه‌بندی
              </h3>
              <p className="mt-1 text-sm text-red-700">
                مشکلی در پردازش متن رخ داده است. لطفاً دوباره تلاش کنید.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClassificationPage;