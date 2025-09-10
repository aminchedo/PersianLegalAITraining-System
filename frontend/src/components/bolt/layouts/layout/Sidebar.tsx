import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Brain,
  Database,
  BarChart3,
  Monitor,
  FileText,
  Settings,
  Users,
  Activity,
  Zap,
} from 'lucide-react';

const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', icon: Brain, label: 'آموزش مدل‌ها', color: 'text-indigo-600' },
    { path: '/dashboard', icon: LayoutDashboard, label: 'داشبورد', color: 'text-purple-600' },
    { path: '/models', icon: Zap, label: 'مدیریت مدل‌ها', color: 'text-blue-600' },
    { path: '/data', icon: Database, label: 'منابع داده', color: 'text-green-600' },
    { path: '/analytics', icon: BarChart3, label: 'آنالیز و گزارش', color: 'text-orange-600' },
    { path: '/monitoring', icon: Monitor, label: 'نظارت سیستم', color: 'text-red-600' },
    { path: '/logs', icon: FileText, label: 'لاگ‌های سیستم', color: 'text-gray-600' },
    { path: '/team', icon: Users, label: 'مدیریت تیم', color: 'text-pink-600' },
    { path: '/settings', icon: Settings, label: 'تنظیمات', color: 'text-gray-500' },
  ];

  return (
    <aside className="bg-white w-64 min-h-screen border-l border-gray-200">
      <div className="p-6">
        <div className="flex items-center space-x-3 space-x-reverse mb-8">
          <div className="bg-gradient-to-br from-purple-600 to-indigo-600 p-2 rounded-lg">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-lg text-gray-900">هوش مصنوعی</h2>
            <p className="text-sm text-gray-500">حقوقی ایران</p>
          </div>
        </div>

        <nav className="space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center space-x-3 space-x-reverse px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-purple-50 text-purple-700 border-r-4 border-purple-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <item.icon className={`w-5 h-5 ${item.color}`} />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="absolute bottom-0 w-64 p-6 bg-gradient-to-t from-purple-50 to-transparent">
        <div className="bg-white rounded-lg p-4 shadow-sm border border-purple-200">
          <div className="flex items-center space-x-3 space-x-reverse mb-2">
            <div className="bg-purple-600 p-1 rounded">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <span className="text-sm font-semibold text-purple-700">وضعیت سیستم</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-600">CPU</span>
              <span className="text-purple-600 font-medium">45%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div className="bg-purple-600 h-1.5 rounded-full" style={{ width: '45%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;