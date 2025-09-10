import React from 'react';
import { Bell, Settings, User, Search, Wifi, WifiOff } from 'lucide-react';
import { usePersianAI } from '../../hooks/usePersianAI';

const Header: React.FC = () => {
  const { state } = usePersianAI();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 space-x-reverse">
          <h1 className="text-2xl font-bold text-gray-900">سامانه آموزش هوش مصنوعی حقوقی</h1>
          <div className="flex items-center">
            {state.isConnected ? (
              <div className="flex items-center text-green-600">
                <Wifi className="w-4 h-4 ml-2" />
                <span className="text-sm">متصل</span>
              </div>
            ) : (
              <div className="flex items-center text-red-600">
                <WifiOff className="w-4 h-4 ml-2" />
                <span className="text-sm">قطع شده</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-4 space-x-reverse">
          <div className="relative">
            <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="جستجو..."
              className="pl-4 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          
          <button className="relative p-2 text-gray-600 hover:text-gray-900 transition-colors">
            <Bell className="w-5 h-5" />
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
              3
            </span>
          </button>
          
          <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors">
            <Settings className="w-5 h-5" />
          </button>
          
          <div className="flex items-center space-x-3 space-x-reverse">
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900">علی احمدی</p>
              <p className="text-xs text-gray-500">مدیر سیستم</p>
            </div>
            <button className="p-2 bg-gray-100 rounded-full hover:bg-gray-200 transition-colors">
              <User className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;