import React, { useState } from 'react';
import { Settings, Database, Globe, Shield, Bell, User, Save, Download, Upload, RefreshCw, Key, Trash2, Plus } from 'lucide-react';

const SettingsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('data-sources');
  const [customUrl, setCustomUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [notifications, setNotifications] = useState({
    training: true,
    system: true,
    email: false,
    slack: true,
  });

  const dataSources = [
    {
      id: 'legal-docs',
      name: 'اسناد حقوقی ایران',
      url: 'https://api.legal-docs.ir/v1',
      type: 'api',
      enabled: true,
      requiresAuth: true,
    },
    {
      id: 'case-studies',
      name: 'آرای قضایی',
      url: 'https://divan.justice.ir/api',
      type: 'api',
      enabled: true,
      requiresAuth: true,
    },
    {
      id: 'regulations',
      name: 'مقررات حقوقی',
      url: 'https://regulations.gov.ir/api',
      type: 'api',
      enabled: false,
      requiresAuth: false,
    },
  ];

  const [customSources, setCustomSources] = useState([
    {
      id: 'custom-1',
      name: 'منبع سفارشی 1',
      url: 'https://custom-legal-api.com/v1',
      type: 'api',
      enabled: true,
    }
  ]);

  const systemSettings = {
    maxConcurrentTraining: 3,
    autoBackup: true,
    retentionDays: 30,
    logLevel: 'info',
    cacheSize: '2GB',
  };

  const addCustomSource = () => {
    if (customUrl) {
      const newSource = {
        id: `custom-${Date.now()}`,
        name: `منبع سفارشی ${customSources.length + 2}`,
        url: customUrl,
        type: 'api',
        enabled: true,
      };
      setCustomSources([...customSources, newSource]);
      setCustomUrl('');
    }
  };

  const removeCustomSource = (id: string) => {
    setCustomSources(customSources.filter(source => source.id !== id));
  };

  const tabs = [
    { id: 'data-sources', name: 'منابع داده', icon: Database },
    { id: 'system', name: 'تنظیمات سیستم', icon: Settings },
    { id: 'security', name: 'امنیت', icon: Shield },
    { id: 'notifications', name: 'اعلان‌ها', icon: Bell },
    { id: 'profile', name: 'پروفایل', icon: User },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-600 to-gray-700 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">تنظیمات سیستم</h1>
            <p className="text-gray-100">پیکربندی منابع داده، سیستم و تنظیمات امنیتی</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <button className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white hover:bg-white/20 transition-colors">
              <Download className="w-5 h-5 ml-2 inline" />
              صدور تنظیمات
            </button>
            <button className="bg-white text-slate-600 px-6 py-2 rounded-lg hover:bg-gray-100 transition-colors font-semibold">
              <Save className="w-5 h-5 ml-2 inline" />
              ذخیره تغییرات
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Tabs */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <nav className="space-y-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 space-x-reverse px-4 py-3 rounded-lg text-right transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-50 text-blue-700 border-r-4 border-blue-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <tab.icon className={`w-5 h-5 ${activeTab === tab.id ? 'text-blue-600' : 'text-gray-400'}`} />
                <span className="font-medium">{tab.name}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-3 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          {/* Data Sources Tab */}
          {activeTab === 'data-sources' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">تنظیمات منابع داده</h2>
                <p className="text-gray-600 mb-6">مدیریت اتصالات به منابع داده و API‌های خارجی</p>
              </div>

              {/* Default Data Sources */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">منابع پیش‌فرض</h3>
                <div className="space-y-4">
                  {dataSources.map((source) => (
                    <div key={source.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3 space-x-reverse">
                          <div className="bg-blue-100 p-2 rounded-lg">
                            <Database className="w-5 h-5 text-blue-600" />
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{source.name}</h4>
                            <p className="text-sm text-gray-600">{source.url}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2 space-x-reverse">
                          <label className="flex items-center">
                            <input
                              type="checkbox"
                              defaultChecked={source.enabled}
                              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="mr-2 text-sm text-gray-700">فعال</span>
                          </label>
                        </div>
                      </div>

                      {source.requiresAuth && (
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">کلید API</label>
                            <input
                              type="password"
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              placeholder="••••••••••••••••"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">نوع احراز هویت</label>
                            <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                              <option value="api-key">API Key</option>
                              <option value="oauth">OAuth 2.0</option>
                              <option value="basic">Basic Auth</option>
                            </select>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Custom Data Sources */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">منابع سفارشی</h3>
                
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 mb-4">
                  <div className="flex items-center space-x-4 space-x-reverse">
                    <div className="flex-1">
                      <input
                        type="url"
                        value={customUrl}
                        onChange={(e) => setCustomUrl(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="https://api.example.com/v1"
                      />
                    </div>
                    <div>
                      <input
                        type="text"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="کلید API (اختیاری)"
                      />
                    </div>
                    <button
                      onClick={addCustomSource}
                      className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      <Plus className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                <div className="space-y-3">
                  {customSources.map((source) => (
                    <div key={source.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3 space-x-reverse">
                        <Globe className="w-5 h-5 text-gray-600" />
                        <div>
                          <p className="font-medium text-gray-900">{source.name}</p>
                          <p className="text-sm text-gray-600">{source.url}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 space-x-reverse">
                        <button className="text-gray-600 hover:text-gray-900 transition-colors">
                          <Settings className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => removeCustomSource(source.id)}
                          className="text-red-600 hover:text-red-700 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Data Pipeline Settings */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">تنظیمات پایپ‌لاین داده</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">روش جمع‌آوری</label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                      <option value="streaming">Stream (زنده)</option>
                      <option value="batch">Batch (دسته‌ای)</option>
                      <option value="hybrid">ترکیبی</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">فواصل بروزرسانی</label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                      <option value="realtime">زنده</option>
                      <option value="hourly">ساعتی</option>
                      <option value="daily">روزانه</option>
                      <option value="weekly">هفتگی</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* System Settings Tab */}
          {activeTab === 'system' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">تنظیمات سیستم</h2>
                <p className="text-gray-600 mb-6">پیکربندی عمومی سیستم و عملکرد</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">حداکثر آموزش همزمان</label>
                  <input
                    type="number"
                    defaultValue={systemSettings.maxConcurrentTraining}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">سطح لاگ</label>
                  <select 
                    defaultValue={systemSettings.logLevel}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="debug">Debug</option>
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">اندازه کش</label>
                  <select 
                    defaultValue={systemSettings.cacheSize}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="1GB">1 گیگابایت</option>
                    <option value="2GB">2 گیگابایت</option>
                    <option value="4GB">4 گیگابایت</option>
                    <option value="8GB">8 گیگابایت</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">مدت نگهداری لاگ (روز)</label>
                  <input
                    type="number"
                    defaultValue={systemSettings.retentionDays}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">پشتیبان‌گیری خودکار</h3>
                    <p className="text-sm text-gray-600">پشتیبان‌گیری روزانه از مدل‌ها و تنظیمات</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      defaultChecked={systemSettings.autoBackup}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>

                <div className="border-t border-gray-200 pt-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                      <RefreshCw className="w-4 h-4 ml-2 inline" />
                      راه‌اندازی مجدد سیستم
                    </button>
                    <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                      <Download className="w-4 h-4 ml-2 inline" />
                      پشتیبان‌گیری دستی
                    </button>
                    <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                      <Upload className="w-4 h-4 ml-2 inline" />
                      بازیابی از پشتیبان
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">تنظیمات امنیتی</h2>
                <p className="text-gray-600 mb-6">مدیریت کلیدها، دسترسی‌ها و امنیت سیستم</p>
              </div>

              <div className="space-y-6">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 space-x-reverse mb-2">
                    <Key className="w-5 h-5 text-yellow-600" />
                    <h3 className="font-semibold text-yellow-800">کلید API اصلی</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <input
                        type="password"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                        defaultValue="sk-1234567890abcdef"
                        readOnly
                      />
                    </div>
                    <button className="bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 transition-colors">
                      تولید کلید جدید
                    </button>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">تنظیمات احراز هویت</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-gray-900">احراز هویت دو مرحله‌ای</h4>
                        <p className="text-sm text-gray-600">افزایش امنیت با کد تأیید</p>
                      </div>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-gray-900">انقضای جلسه</h4>
                        <p className="text-sm text-gray-600">مدت زمان فعالیت جلسه کاری</p>
                      </div>
                      <select className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option value="30">30 دقیقه</option>
                        <option value="60">1 ساعت</option>
                        <option value="240">4 ساعت</option>
                        <option value="480">8 ساعت</option>
                      </select>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium text-gray-900">لاگ دسترسی</h4>
                        <p className="text-sm text-gray-600">ثبت تمام فعالیت‌های کاربران</p>
                      </div>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          defaultChecked={true}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                      </label>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">آدرس‌های IP مجاز</h3>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 space-x-reverse">
                      <input
                        type="text"
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="192.168.1.0/24"
                      />
                      <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                        <Plus className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="space-y-2 mt-4">
                      <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-700">192.168.1.0/24</span>
                        <button className="text-red-600 hover:text-red-700">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Notifications Tab */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">تنظیمات اعلان‌ها</h2>
                <p className="text-gray-600 mb-6">مدیریت اعلان‌ها و هشدارهای سیستم</p>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">اعلان‌های آموزش</h3>
                    <p className="text-sm text-gray-600">اطلاع‌رسانی شروع، پایان و خطاهای آموزش</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={notifications.training}
                      onChange={(e) => setNotifications({...notifications, training: e.target.checked})}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">اعلان‌های سیستم</h3>
                    <p className="text-sm text-gray-600">هشدارهای مربوط به وضعیت سیستم و منابع</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={notifications.system}
                      onChange={(e) => setNotifications({...notifications, system: e.target.checked})}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">ایمیل</h3>
                    <p className="text-sm text-gray-600">ارسال اعلان‌ها از طریق ایمیل</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={notifications.email}
                      onChange={(e) => setNotifications({...notifications, email: e.target.checked})}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">Slack</h3>
                    <p className="text-sm text-gray-600">ارسال اعلان‌ها به کانال Slack</p>
                  </div>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={notifications.slack}
                      onChange={(e) => setNotifications({...notifications, slack: e.target.checked})}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                  </label>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">تنظیمات پیشرفته</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">حد آستانه CPU (%)</label>
                    <input
                      type="number"
                      defaultValue="80"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">حد آستانه حافظه (%)</label>
                    <input
                      type="number"
                      defaultValue="85"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">پروفایل کاربری</h2>
                <p className="text-gray-600 mb-6">مدیریت اطلاعات شخصی و تنظیمات حساب</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">نام کامل</label>
                  <input
                    type="text"
                    defaultValue="علی احمدی"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">ایمیل</label>
                  <input
                    type="email"
                    defaultValue="admin@example.com"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">شماره تلفن</label>
                  <input
                    type="tel"
                    defaultValue="09121234567"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">زبان رابط کاربری</label>
                  <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    <option value="fa">فارسی</option>
                    <option value="en">انگلیسی</option>
                  </select>
                </div>
              </div>

              <div className="border-t border-gray-200 pt-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">تغییر رمز عبور</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">رمز عبور فعلی</label>
                    <input
                      type="password"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div></div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">رمز عبور جدید</label>
                    <input
                      type="password"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">تکرار رمز عبور جدید</label>
                    <input
                      type="password"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
                <button className="mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                  تغییر رمز عبور
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;