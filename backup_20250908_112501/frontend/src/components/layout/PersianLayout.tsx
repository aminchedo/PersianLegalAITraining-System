import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Search, FileText, Brain, Activity, Settings,
  Menu, X, Home, Database
} from 'lucide-react';

interface PersianLayoutProps {
  children: React.ReactNode;
}

const PersianLayout: React.FC<PersianLayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'خانه', href: '/', icon: Home },
    { name: 'اسناد', href: '/documents', icon: FileText },
    { name: 'آموزش', href: '/training', icon: Brain },
    { name: 'طبقه‌بندی', href: '/classification', icon: Database },
    { name: 'سیستم', href: '/system', icon: Activity },
  ];

  return (
    <div className="min-h-screen bg-gray-50 font-vazir">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed top-0 right-0 h-full w-64 bg-blue-900 text-white transform transition-transform duration-300 z-50 ${
        sidebarOpen ? 'translate-x-0' : 'translate-x-full'
      } lg:translate-x-0`}>
        <div className="flex items-center justify-between p-4 border-b border-blue-800">
          <h1 className="text-xl font-bold">سیستم حقوقی هوشمند</h1>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden text-white hover:text-gray-300"
          >
            <X size={24} />
          </button>
        </div>

        <nav className="mt-8">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-6 py-3 text-right hover:bg-blue-800 transition-colors ${
                  isActive ? 'bg-blue-800 border-l-4 border-white' : ''
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                <item.icon className="ml-3" size={20} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* System status */}
        <div className="absolute bottom-4 right-4 left-4">
          <div className="bg-blue-800 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <span>وضعیت سیستم</span>
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:mr-64">
        {/* Top navigation */}
        <header className="bg-white shadow-sm border-b">
          <div className="flex items-center justify-between px-6 py-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden text-gray-600 hover:text-gray-900"
            >
              <Menu size={24} />
            </button>
            
            <div className="flex items-center space-x-4 space-x-reverse">
              <span className="text-sm text-gray-600">
                سیستم آموزش هوش مصنوعی حقوقی فارسی
              </span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

export default PersianLayout;