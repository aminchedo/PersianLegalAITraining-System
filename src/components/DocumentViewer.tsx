import React from 'react';
import { X, ExternalLink, Tag, Clock, FileText, Sparkles } from 'lucide-react';
import { useDocument, useClassifyDocument } from '../hooks/useDocuments';

interface DocumentViewerProps {
  documentId: number;
  onClose: () => void;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({ documentId, onClose }) => {
  const { data: document, isLoading, error } = useDocument(documentId);
  const classifyMutation = useClassifyDocument();

  const handleClassify = () => {
    classifyMutation.mutate(documentId);
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('fa-IR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'تاریخ نامشخص';
    }
  };

  const getCategoryLabel = (category: string) => {
    const categories: Record<string, string> = {
      'civil_law': 'حقوق مدنی',
      'criminal_law': 'حقوق جزا',
      'commercial_law': 'حقوق تجارت',
      'administrative_law': 'حقوق اداری',
      'constitutional_law': 'حقوق قانون اساسی',
      'labor_law': 'حقوق کار',
      'family_law': 'حقوق خانواده',
      'property_law': 'حقوق اموال',
      'tax_law': 'حقوق مالیاتی',
      'international_law': 'حقوق بین‌الملل',
      'other': 'سایر'
    };
    return categories[category] || category;
  };

  if (error) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-red-600">خطا</h2>
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 rounded-full"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <p className="text-gray-600">خطا در بارگذاری سند. لطفاً دوباره تلاش کنید.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">نمایش سند</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">در حال بارگذاری سند...</p>
            </div>
          </div>
        )}

        {/* Document Content */}
        {document && (
          <div className="flex-1 overflow-y-auto">
            {/* Document Info */}
            <div className="p-6 border-b bg-gray-50">
              <h1 className="text-2xl font-bold text-gray-900 mb-4">
                {document.title}
              </h1>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-600">تاریخ جمع‌آوری:</span>
                  <span className="font-medium">{formatDate(document.scraped_at)}</span>
                </div>
                
                <div className="flex items-center gap-2">
                  <ExternalLink className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-600">منبع:</span>
                  <span className="font-medium">{document.source}</span>
                </div>
                
                {document.category && (
                  <div className="flex items-center gap-2">
                    <Tag className="w-4 h-4 text-gray-500" />
                    <span className="text-gray-600">دسته‌بندی:</span>
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">
                      {getCategoryLabel(document.category)}
                    </span>
                  </div>
                )}
              </div>
              
              {/* Action Buttons */}
              <div className="flex gap-3 mt-4">
                <button
                  onClick={handleClassify}
                  disabled={classifyMutation.isPending}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Sparkles className="w-4 h-4" />
                  {classifyMutation.isPending ? 'در حال دسته‌بندی...' : 'دسته‌بندی هوشمند'}
                </button>
                
                <a
                  href={document.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  مشاهده در سایت اصلی
                </a>
              </div>
            </div>

            {/* Document Text */}
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">متن سند</h3>
              <div 
                className="prose prose-lg max-w-none text-gray-800 leading-relaxed"
                style={{ direction: 'rtl', textAlign: 'right' }}
              >
                {document.content.split('\n').map((paragraph, index) => (
                  <p key={index} className="mb-4">
                    {paragraph}
                  </p>
                ))}
              </div>
            </div>

            {/* Classification Results */}
            {classifyMutation.data && (
              <div className="p-6 border-t bg-green-50">
                <h3 className="text-lg font-semibold text-green-900 mb-4">
                  نتیجه دسته‌بندی هوشمند
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-green-700">دسته‌بندی:</span>
                    <div className="bg-green-100 text-green-800 px-3 py-2 rounded-lg font-medium mt-1">
                      {getCategoryLabel(classifyMutation.data.category)}
                    </div>
                  </div>
                  <div>
                    <span className="text-sm text-green-700">اعتماد:</span>
                    <div className="bg-green-100 text-green-800 px-3 py-2 rounded-lg font-medium mt-1">
                      {(classifyMutation.data.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Classification Error */}
            {classifyMutation.error && (
              <div className="p-6 border-t bg-red-50">
                <p className="text-red-600">
                  خطا در دسته‌بندی هوشمند. لطفاً دوباره تلاش کنید.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;