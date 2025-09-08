import React, { useState, useCallback, useEffect } from 'react';
import { Search, Filter, FileText, Clock, Tag, ExternalLink } from 'lucide-react';
import { useDocuments } from '../hooks/useDocuments';
import type { Document } from '../types';
import { debounce } from 'lodash';

interface SearchInterfaceProps {
  onDocumentSelect: (id: number) => void;
  selectedDocumentId?: number | null;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ 
  onDocumentSelect, 
  selectedDocumentId 
}) => {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<string>('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Debounce search query
  const debouncedSetQuery = useCallback(
    debounce((value: string) => {
      setDebouncedQuery(value);
    }, 500),
    []
  );

  useEffect(() => {
    debouncedSetQuery(query);
    return () => {
      debouncedSetQuery.cancel();
    };
  }, [query, debouncedSetQuery]);

  const { data: searchResults, isLoading, error } = useDocuments(
    debouncedQuery.length >= 2 ? debouncedQuery : undefined,
    category || undefined
  );

  const categories = [
    { value: '', label: 'همه دسته‌ها' },
    { value: 'civil_law', label: 'حقوق مدنی' },
    { value: 'criminal_law', label: 'حقوق جزا' },
    { value: 'commercial_law', label: 'حقوق تجارت' },
    { value: 'administrative_law', label: 'حقوق اداری' },
    { value: 'constitutional_law', label: 'حقوق قانون اساسی' },
    { value: 'labor_law', label: 'حقوق کار' },
    { value: 'family_law', label: 'حقوق خانواده' },
    { value: 'property_law', label: 'حقوق اموال' },
    { value: 'tax_law', label: 'حقوق مالیاتی' },
    { value: 'international_law', label: 'حقوق بین‌الملل' },
  ];

  const handleDocumentClick = (document: Document) => {
    onDocumentSelect(document.id);
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('fa-IR');
    } catch {
      return 'تاریخ نامشخص';
    }
  };

  const truncateText = (text: string, maxLength: number = 200) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col gap-4">
          {/* Search Input */}
          <div className="relative">
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="جستجو در اسناد حقوقی..."
              className="block w-full pr-10 pl-3 py-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-lg"
              dir="rtl"
            />
          </div>

          {/* Filters Toggle */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <Filter className="w-4 h-4" />
              فیلترها
              {category && (
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                  فعال
                </span>
              )}
            </button>
            
            {searchResults && (
              <span className="text-sm text-gray-600">
                {searchResults.total} نتیجه یافت شد
              </span>
            )}
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="border-t pt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    دسته‌بندی
                  </label>
                  <select
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {categories.map((cat) => (
                      <option key={cat.value} value={cat.value}>
                        {cat.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Search Results */}
      <div className="bg-white rounded-lg shadow-md">
        {/* Loading State */}
        {isLoading && (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">در حال جستجو...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="p-8 text-center">
            <p className="text-red-600">خطا در جستجو. لطفاً دوباره تلاش کنید.</p>
          </div>
        )}

        {/* No Query State */}
        {!debouncedQuery && !isLoading && (
          <div className="p-8 text-center">
            <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              جستجوی اسناد حقوقی
            </h3>
            <p className="text-gray-600">
              برای شروع، کلمه کلیدی مورد نظر خود را وارد کنید
            </p>
          </div>
        )}

        {/* No Results State */}
        {debouncedQuery && searchResults && searchResults.documents.length === 0 && !isLoading && (
          <div className="p-8 text-center">
            <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              نتیجه‌ای یافت نشد
            </h3>
            <p className="text-gray-600">
              برای "{debouncedQuery}" هیچ سندی پیدا نشد. کلمات کلیدی دیگری امتحان کنید.
            </p>
          </div>
        )}

        {/* Results List */}
        {searchResults && searchResults.documents.length > 0 && (
          <div className="divide-y divide-gray-200">
            {searchResults.documents.map((document) => (
              <div
                key={document.id}
                onClick={() => handleDocumentClick(document)}
                className={`p-6 hover:bg-gray-50 cursor-pointer transition-colors ${
                  selectedDocumentId === document.id ? 'bg-blue-50 border-r-4 border-blue-500' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-medium text-gray-900 mb-2 line-clamp-2">
                      {document.title}
                    </h3>
                    
                    {/* Document snippet */}
                    <p className="text-gray-600 mb-3 line-clamp-3">
                      {truncateText(document.content)}
                    </p>
                    
                    {/* Document metadata */}
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <div className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {formatDate(document.scraped_at)}
                      </div>
                      
                      {document.category && (
                        <div className="flex items-center gap-1">
                          <Tag className="w-4 h-4" />
                          {categories.find(c => c.value === document.category)?.label || document.category}
                        </div>
                      )}
                      
                      <div className="flex items-center gap-1">
                        <ExternalLink className="w-4 h-4" />
                        {document.source}
                      </div>
                    </div>
                  </div>
                  
                  <div className="mr-4 flex-shrink-0">
                    <FileText className="w-6 h-6 text-gray-400" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Load More Button */}
      {searchResults && searchResults.documents.length > 0 && searchResults.documents.length < searchResults.total && (
        <div className="text-center">
          <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            نمایش نتایج بیشتر
          </button>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;