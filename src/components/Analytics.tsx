import React from 'react';
import { BarChart3, PieChart, TrendingUp, FileText, Clock, Tag } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import apiService from '../services/apiService';

interface DocumentStats {
  total_documents: number;
  documents_by_category: Record<string, number>;
  documents_by_source: Record<string, number>;
  recent_scraping_activity: Array<{
    date: string;
    documents_added: number;
  }>;
}

interface CategoryStat {
  category: string;
  count: number;
  avg_confidence: number;
  first_document: string;
  last_document: string;
}

const Analytics: React.FC = () => {
  const { data: stats, isLoading: statsLoading } = useQuery<DocumentStats>({
    queryKey: ['documentStats'],
    queryFn: () => apiService.getDocumentStats(),
  });

  const { data: categoryStats, isLoading: categoryLoading } = useQuery<CategoryStat[]>({
    queryKey: ['categoryStats'],
    queryFn: () => apiService.getCategoryStats(),
  });

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

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('fa-IR');
    } catch {
      return 'تاریخ نامشخص';
    }
  };

  if (statsLoading || categoryLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="space-y-3">
              <div className="h-4 bg-gray-200 rounded"></div>
              <div className="h-4 bg-gray-200 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const totalRecentActivity = stats?.recent_scraping_activity?.reduce((sum, day) => sum + day.documents_added, 0) || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center gap-3 mb-4">
          <BarChart3 className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">تحلیل و آمار</h2>
        </div>
        <p className="text-gray-600">
          آمار کلی از اسناد جمع‌آوری شده و وضعیت سامانه
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">کل اسناد</p>
              <p className="text-2xl font-bold text-blue-600">
                {stats?.total_documents?.toLocaleString('fa-IR') || '0'}
              </p>
            </div>
            <FileText className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">دسته‌بندی‌ها</p>
              <p className="text-2xl font-bold text-green-600">
                {Object.keys(stats?.documents_by_category || {}).length}
              </p>
            </div>
            <Tag className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">منابع</p>
              <p className="text-2xl font-bold text-purple-600">
                {Object.keys(stats?.documents_by_source || {}).length}
              </p>
            </div>
            <PieChart className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">فعالیت اخیر</p>
              <p className="text-2xl font-bold text-orange-600">
                {totalRecentActivity}
              </p>
              <p className="text-xs text-gray-500">۷ روز گذشته</p>
            </div>
            <TrendingUp className="w-8 h-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* Documents by Category */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            توزیع بر اساس دسته‌بندی
          </h3>
          <div className="space-y-3">
            {Object.entries(stats?.documents_by_category || {})
              .sort(([,a], [,b]) => (b as number) - (a as number))
              .slice(0, 8)
              .map(([category, count]) => {
                const total = stats?.total_documents || 1;
                const percentage = ((count as number) / total) * 100;
                
                return (
                  <div key={category} className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">
                          {getCategoryLabel(category)}
                        </span>
                        <span className="text-sm text-gray-500">
                          {(count as number).toLocaleString('fa-IR')} ({percentage.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Documents by Source */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            توزیع بر اساس منبع
          </h3>
          <div className="space-y-3">
            {Object.entries(stats?.documents_by_source || {})
              .sort(([,a], [,b]) => (b as number) - (a as number))
              .map(([source, count]) => {
                const total = stats?.total_documents || 1;
                const percentage = ((count as number) / total) * 100;
                
                return (
                  <div key={source} className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-700">
                          {source}
                        </span>
                        <span className="text-sm text-gray-500">
                          {(count as number).toLocaleString('fa-IR')} ({percentage.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          فعالیت اخیر (۷ روز گذشته)
        </h3>
        {stats?.recent_scraping_activity && stats.recent_scraping_activity.length > 0 ? (
          <div className="space-y-3">
            {stats.recent_scraping_activity.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Clock className="w-4 h-4 text-gray-500" />
                  <span className="font-medium text-gray-900">
                    {formatDate(activity.date)}
                  </span>
                </div>
                <span className="text-sm text-gray-600">
                  {activity.documents_added.toLocaleString('fa-IR')} سند جدید
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Clock className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">فعالیت اخیری ثبت نشده است</p>
          </div>
        )}
      </div>

      {/* Category Details */}
      {categoryStats && categoryStats.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            جزئیات دسته‌بندی‌ها
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    دسته‌بندی
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    تعداد اسناد
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    میانگین اعتماد
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    آخرین سند
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {categoryStats.map((category, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {getCategoryLabel(category.category)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {category.count.toLocaleString('fa-IR')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {(category.avg_confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(category.last_document)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;