  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      {/* Full Screen Chart Modal */}
      {fullScreenChart && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-7xl h-5/6 p-6 relative">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">
                {fullScreenChart === 'system-performance' ? 'عملکرد سیستم - نمای کامل' : 'پیشرفت آموزش - نمای کامل'}
              </h2>
              <button 
                onClick={() => setFullScreenChart(null)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-all"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="h-full pb-16">
              <ResponsiveContainer width="100%" height="100%">
                {fullScreenChart === 'system-performance' ? (
                  <AreaChart data={realTimeData}>
                    <defs>
                      <linearGradient id="cpuGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="memoryGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: 'none', 
                        borderRadius: '12px', 
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' 
                      }}
                    />
                    <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGradientFull)" strokeWidth={3} />
                    <Area type="monotone" dataKey="memory" stroke="#10B981" fill="url(#memoryGradientFull)" strokeWidth={3} />
                  </AreaChart>
                ) : (
                  <LineChart data={realTimeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: 'none', 
                        borderRadius: '12px', 
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' 
                      }}
                    />
                    <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={4} dot={false} />
                    <Line type="monotone" dataKey="accuracy" stroke="#8B5CF6" strokeWidth={4} dot={false} />
                    <Line type="monotone" dataKey="throughput" stroke="#F59E0B" strokeWidth={3} dot={false} />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`fixed top-0 right-0 h-full bg-white shadow-2xl border-l border-gray-200 transition-all duration-300 z-40 ${
        sidebarCollapsed ? 'w-20' : 'w-64'
      }`}>
        <div className="p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            {!sidebarCollapsed && (
              <div>
                <h1 className="font-bold text-gray-900">AI حقوقی</h1>
                <p className="text-xs text-gray-500">فارسی ۲۰۲۵</p>
              </div>
            )}
          </div>
        </div>

        <nav className="px-4 space-y-2">
          <MenuItem icon={Home} label="داشبورد" id="dashboard" />
          <MenuItem icon={Brain} label="مدل‌ها" id="models" badge="4" />
          <MenuItem icon={Database} label="داده‌ها" id="data" />
          <MenuItem icon={Activity} label="نظارت" id="monitoring" />
          <MenuItem icon={FileText} label="گزارش‌ها" id="reports" />
          <MenuItem icon={Users} label="تیم" id="team" />
          <MenuItem icon={Settings} label="تنظیمات" id="settings" />
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <div className="text-center mb-4">
            {!sidebarCollapsed && (
              <div className="bg-gradient-to-r from-green-100 to-blue-100 rounded-xl p-3">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  سیستم فعال
                </div>
                <p className="text-xs text-gray-500 mt-1">24/7 در حال کار</p>
              </div>
            )}
          </div>
          <button 
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center py-2 text-gray-500 hover:text-gray-700 transition-all"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className={`transition-all duration-300 ${sidebarCollapsed ? 'mr-20' : 'mr-64'}`}>
        {/* Enhanced Top Bar */}
        <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="w-5 h-5 text-gray-400 absolute right-3 top-1/2 transform -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="جستجو در مدل‌ها، داده‌ها و گزارش‌ها..."
                className="w-96 pl-4 pr-10 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-xl text-gray-600 hover:bg-gray-200 transition-all">
              <Filter className="w-4 h-4" />
              فیلتر
            </button>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-gray-100 rounded-xl p-1">
              <button 
                onClick={() => setRefreshInterval(1000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 1000 ? 'bg-white shadow-sm' : ''}`}
              >
                1s
              </button>
              <button 
                onClick={() => setRefreshInterval(3000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 3000 ? 'bg-white shadow-sm' : ''}`}
              >
                3s
              </button>
              <button 
                onClick={() => setRefreshInterval(5000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 5000 ? 'bg-white shadow-sm' : ''}`}
              >
                5s
              </button>
            </div>
            
            <div className="relative">
              <button 
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-xl transition-all"
              >
                <Bell className="w-5 h-5" />
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-xs">{notifications.length}</span>
                </span>
              </button>
              <NotificationPanel />
            </div>
            
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center cursor-pointer hover:shadow-lg transition-all">
              <span className="text-white text-sm font-medium">ک</span>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <div className="p-6">
          {activeTab === 'dashboard' && renderDashboard()}
          {activeTab === 'models' && renderModels()}
          {activeTab === 'settings' && renderSettings()}
          {activeTab === 'data' && <div className="text-center py-20 text-gray-500">صفحه داده‌ها در حال توسعه...</div>}
          {activeTab === 'monitoring' && <div className="text-center py-20 text-gray-500">صفحه نظارت در حال توسعه...</div>}
          {activeTab === 'reports' && <div className="text-center py-20 text-gray-500">صفحه گزارش‌ها در حال توسعه...</div>}
          {activeTab === 'team' && <div className="text-center py-20 text-gray-500">صفحه تیم در حال توسعه...</div>}
        </div>
      </div>
    </div>
  );