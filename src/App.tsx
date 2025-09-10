import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Search, FileText, BarChart3, Settings, BookOpen, Server } from 'lucide-react';
import SearchInterface from './components/SearchInterface';
import ScrapingStatus from './components/ScrapingStatus';
import DocumentViewer from './components/DocumentViewer';
import Analytics from './components/Analytics';
import DeploymentMonitor from './components/DeploymentMonitor';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

type TabType = 'search' | 'scraping' | 'analytics' | 'deployment' | 'settings';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('search');
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null);

  const tabs = [
    { id: 'search' as TabType, name: 'جستجو', icon: Search },
    { id: 'scraping' as TabType, name: 'جمع‌آوری', icon: FileText },
    { id: 'analytics' as TabType, name: 'تحلیل', icon: BarChart3 },
    { id: 'deployment' as TabType, name: 'استقرار', icon: Server },
    { id: 'settings' as TabType, name: 'تنظیمات', icon: Settings },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'search':
        return (
          <SearchInterface 
            onDocumentSelect={setSelectedDocumentId}
            selectedDocumentId={selectedDocumentId}
          />
        );
      case 'scraping':
        return <ScrapingStatus />;
      case 'analytics':
        return <Analytics />;
      case 'deployment':
        return <DeploymentMonitor />;
      case 'settings':
        return <div className="p-6">تنظیمات در حال توسعه...</div>;
      default:
        return <SearchInterface onDocumentSelect={setSelectedDocumentId} />;
    }
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50" dir="rtl">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center gap-3">
                <BookOpen className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">
                    آرشیو اسناد حقوقی ایران
                  </h1>
                  <p className="text-sm text-gray-600">
                    سامانه هوشمند جمع‌آوری و جستجوی اسناد حقوقی
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <span>نسخه 1.0.0</span>
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav className="bg-white border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8 space-x-reverse">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {tab.name}
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {renderContent()}
        </main>

        {/* Document Viewer Modal */}
        {selectedDocumentId && (
          <DocumentViewer
            documentId={selectedDocumentId}
            onClose={() => setSelectedDocumentId(null)}
          />
        )}
      </div>

      {/* React Query Devtools */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;