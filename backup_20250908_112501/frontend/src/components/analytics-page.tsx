import React, { useState, useEffect } from 'react';
import { 
  BarChart3, TrendingUp, TrendingDown, PieChart, LineChart, 
  Calendar, Filter, Download, RefreshCw, Eye, Settings,
  Brain, Database, Activity, Clock, Users, Target,
  ArrowUp, ArrowDown, Minus, Plus, X, Maximize2
} from 'lucide-react';
import { 
  BarChart, Bar, LineChart as ReLineChart, Line, AreaChart, Area, 
  PieChart as RePieChart, Pie, Cell, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, ComposedChart
} from 'recharts';
import { useAppContext } from './Router';

const AnalyticsPage = () => {
  const { models, dataSources, realTimeData, projects } = useAppContext();
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const [selectedProject, setSelectedProject] = useState('all');
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [fullScreenChart, setFullScreenChart] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Generate analytics data
  const [analyticsData, setAnalyticsData] = useState({
    modelPerformance: [],
    dataQualityTrends: [],
    trainingMetrics: [],
    resourceUtilization: [],
    accuracyTrends: [],
    errorRates: [],
    throughputData: []
  });

  useEffect(() => {
    // Load real analytics data
    const loadAnalytics = () => {
      const days = selectedTimeRange === '7d' ? 7 : selectedTimeRange === '30d' ? 30 : 90;
      
      const modelPerformance = Array.from({ length: days }, (_, i) => ({
        date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('fa-IR'),
        accuracy: 85 + Math.random() * 10 + (i / days) * 5,
        loss: Math.exp(-i/20) * (0.5 + Math.random() * 0.3),
        f1Score: 80 + Math.random() * 15 + (i / days) * 3,
        precision: 82 + Math.random() * 12 + (i / days) * 4,
        recall: 79 + Math.random() * 13 + (i / days) * 5
      }));

      const dataQualityTrends = Array.from({ length: days }, (_, i) => ({
        date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('fa-IR'),
        quality: 85 + Math.random() * 10,
        coverage: 90 + Math.random() * 8,
        consistency: 88 + Math.random() * 9,
        completeness: 92 + Math.random() * 6
      }));

      const trainingMetrics = Array.from({ length: 50 }, (_, i) => ({
        epoch: i + 1,
        trainLoss: Math.exp(-i/30) * (1 + Math.random() * 0.3),
        valLoss: Math.exp(-i/25) * (1.1 + Math.random() * 0.3),
        trainAcc: 60 + i * 0.8 + Math.random() * 3,
        valAcc: 58 + i * 0.7 + Math.random() * 4,
        learningRate: 1e-4 * Math.exp(-i/40)
      }));

      const resourceUtilization = Array.from({ length: 24 }, (_, i) => ({
        hour: `${i}:00`,
        cpuUsage: 40 + Math.random() * 40,
        memoryUsage: 50 + Math.random() * 30,
        gpuUsage: 30 + Math.random() * 50,
        diskIO: 20 + Math.random() * 60
      }));

      setAnalyticsData({
        modelPerformance,
        dataQualityTrends,
        trainingMetrics,
        resourceUtilization,
        accuracyTrends: modelPerformance,
        errorRates: modelPerformance.map(item => ({ ...item, errorRate: 100 - item.accuracy })),
        throughputData: Array.from({ length: days }, (_, i) => ({
          date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('fa-IR'),
          throughput: 80 + Math.random() * 40,
          batchSize: Math.floor(Math.random() * 8) + 2,
          processedSamples: Math.floor(Math.random() * 10000) + 5000
        }))
      });
    };

    generateAnalytics();
  }, [selectedTimeRange, selectedProject]);

  // Calculate KPIs
  const calculateKPIs = () => {
    const currentData = analyticsData.modelPerformance;
    if (currentData.length === 0) return {};

    const latest = currentData[currentData.length - 1];
    const previous = currentData.length > 7 ? currentData[currentData.length - 8] : currentData[0];

    return {
      avgAccuracy: {
        current: latest?.accuracy || 0,
        change: ((latest?.accuracy || 0) - (previous?.accuracy || 0)),
        trend: (latest?.accuracy || 0) > (previous?.accuracy || 0) ? 'up' : 'down'
      },
      avgLoss: {
        current: latest?.loss || 0,
        change: ((latest?.loss || 0) - (previous?.loss || 0)),
        trend: (latest?.loss || 0) < (previous?.loss || 0) ? 'up' : 'down'
      },
      dataQuality: {
        current: analyticsData.dataQualityTrends[analyticsData.dataQualityTrends.length - 1]?.quality || 0,
        change: 2.3,
        trend: 'up'
      },
      throughput: {
        current: analyticsData.throughputData[analyticsData.throughputData.length - 1]?.throughput || 0,
        change: 15.2,
        trend: 'up'
      }
    };
  };

  const kpis = calculateKPIs();

  // Model comparison data
  const modelComparisonData = models.map(model => ({
    name: model.name,
    accuracy: model.accuracy,
    f1Score: model.accuracy - 2 + Math.random() * 4,
    precision: model.accuracy - 1 + Math.random() * 3,
    recall: model.accuracy - 3 + Math.random() * 5,
    status: model.status
  }));

  // Project performance data
  const projectPerformanceData = projects.map(project => ({
    name: project.name,
    progress: project.progress,
    modelsCount: Math.floor(Math.random() * 5) + 1,
    accuracy: 85 + Math.random() * 10,
    dataSize: Math.floor(Math.random() * 50000) + 10000
  }));

  // Chart colors
  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

  const KPICard = ({ title, value, unit, change, trend, icon: Icon, color }) => (
    <div className={`bg-white rounded-2xl p-6 shadow-lg border-l-4 border-${color}-500 hover:shadow-xl transition-all duration-300`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-600 text-sm font-medium">{title}</p>
          <div className="flex items-baseline gap-2 mb-2">
            <h3 className="text-3xl font-bold text-gray-900">
              {typeof value === 'number' ? value.toFixed(unit === '%' ? 1 : 3) : value}
            </h3>
            <span className="text-gray-500 text-sm">{unit}</span>
          </div>
          <div className="flex items-center gap-1">
            {trend === 'up' ? (
              <ArrowUp className="w-4 h-4 text-green-500" />
            ) : trend === 'down' ? (
              <ArrowDown className="w-4 h-4 text-red-500" />
            ) : (
              <Minus className="w-4 h-4 text-gray-400" />
            )}
            <span className={`text-sm font-medium ${
              trend === 'up' ? 'text-green-600' : 
              trend === 'down' ? 'text-red-600' : 'text-gray-500'
            }`}>
              {Math.abs(change).toFixed(1)}
            </span>
            <span className="text-gray-500 text-sm">از هفته گذشته</span>
          </div>
        </div>
        <div className={`bg-${color}-100 rounded-xl p-3`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <BarChart3 className="w-8 h-8" />
              آنالیز و گزارش‌گیری
            </h1>
            <p className="text-purple-100">تحلیل عملکرد، روند پیشرفت و بینش‌های هوشمند</p>
          </div>
          <div className="flex items-center gap-3">
            <select 
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value)}
              className="bg-white/20 text-white border border-white/30 rounded-xl px-4 py-2 backdrop-blur-sm"
            >
              <option value="7d" className="text-gray-900">7 روز</option>
              <option value="30d" className="text-gray-900">30 روز</option>
              <option value="90d" className="text-gray-900">90 روز</option>
            </select>
            <select 
              value={selectedProject}
              onChange={(e) => setSelectedProject(e.target.value)}
              className="bg-white/20 text-white border border-white/30 rounded-xl px-4 py-2 backdrop-blur-sm"
            >
              <option value="all" className="text-gray-900">همه پروژه‌ها</option>
              {projects.map(project => (
                <option key={project.id} value={project.id} className="text-gray-900">
                  {project.name}
                </option>
              ))}
            </select>
            <button className="bg-white text-indigo-600 px-4 py-2 rounded-xl hover:bg-indigo-50 transition-all flex items-center gap-2">
              <Download className="w-4 h-4" />
              گزارش
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-2xl p-2 shadow-lg">
        <div className="flex gap-2">
          {[
            { id: 'overview', label: 'نمای کلی', icon: BarChart3 },
            { id: 'models', label: 'مدل‌ها', icon: Brain },
            { id: 'data', label: 'داده‌ها', icon: Database },
            { id: 'performance', label: 'عملکرد', icon: Activity },
            { id: 'trends', label: 'روندها', icon: TrendingUp }
          ].map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                  activeTab === tab.id 
                    ? 'bg-indigo-600 text-white shadow-md' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <KPICard 
              title="میانگین دقت"
              value={kpis.avgAccuracy?.current}
              unit="%"
              change={kpis.avgAccuracy?.change}
              trend={kpis.avgAccuracy?.trend}
              icon={Target}
              color="blue"
            />
            <KPICard 
              title="میانگین خطا"
              value={kpis.avgLoss?.current}
              unit=""
              change={kpis.avgLoss?.change}
              trend={kpis.avgLoss?.trend}
              icon={TrendingDown}
              color="red"
            />
            <KPICard 
              title="کیفیت داده"
              value={kpis.dataQuality?.current}
              unit="%"
              change={kpis.dataQuality?.change}
              trend={kpis.dataQuality?.trend}
              icon={Database}
              color="green"
            />
            <KPICard 
              title="نرخ پردازش"
              value={kpis.throughput?.current}
              unit="نمونه/ثانیه"
              change={kpis.throughput?.change}
              trend={kpis.throughput?.trend}
              icon={Activity}
              color="purple"
            />
          </div>

          {/* Main Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Model Performance Trend */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-900">روند عملکرد مدل‌ها</h3>
                <button 
                  onClick={() => setFullScreenChart('performance')}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <ReLineChart data={analyticsData.modelPerformance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="f1Score" stroke="#10B981" strokeWidth={2} dot={false} />
                </ReLineChart>
              </ResponsiveContainer>
              <div className="flex items-center gap-6 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>دقت</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>F1-Score</span>
                </div>
              </div>
            </div>

            {/* Data Quality Trends */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-900">روند کیفیت داده</h3>
                <button 
                  onClick={() => setFullScreenChart('data-quality')}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analyticsData.dataQualityTrends}>
                  <defs>
                    <linearGradient id="qualityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Area type="monotone" dataKey="quality" stroke="#10B981" fill="url(#qualityGradient)" strokeWidth={2} />
                  <Area type="monotone" dataKey="consistency" stroke="#3B82F6" fill="transparent" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model Comparison */}
          <div className="bg-white rounded-2xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">مقایسه عملکرد مدل‌ها</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1Score" fill="#10B981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="precision" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Training Progress */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">پیشرفت آموزش</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ReLineChart data={analyticsData.trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="epoch" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Line type="monotone" dataKey="trainLoss" stroke="#EF4444" strokeWidth={2} />
                  <Line type="monotone" dataKey="valLoss" stroke="#F59E0B" strokeWidth={2} />
                  <Line type="monotone" dataKey="trainAcc" stroke="#3B82F6" strokeWidth={2} />
                  <Line type="monotone" dataKey="valAcc" stroke="#10B981" strokeWidth={2} />
                </ReLineChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>خطای آموزش</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>خطای اعتبارسنجی</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>دقت آموزش</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>دقت اعتبارسنجی</span>
                </div>
              </div>
            </div>

            {/* Model Status Distribution */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">توزیع وضعیت مدل‌ها</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RePieChart>
                  <Pie
                    data={[
                      { name: 'در حال آموزش', value: models.filter(m => m.status === 'training').length, fill: '#3B82F6' },
                      { name: 'تکمیل شده', value: models.filter(m => m.status === 'completed').length, fill: '#10B981' },
                      { name: 'در انتظار', value: models.filter(m => m.status === 'pending').length, fill: '#F59E0B' },
                      { name: 'خطا', value: models.filter(m => m.status === 'error').length, fill: '#EF4444' }
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="value"
                  >
                  </Pie>
                  <Tooltip />
                </RePieChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-2 gap-2 mt-4">
                {[
                  { name: 'در حال آموزش', color: '#3B82F6' },
                  { name: 'تکمیل شده', color: '#10B981' },
                  { name: 'در انتظار', color: '#F59E0B' },
                  { name: 'خطا', color: '#EF4444' }
                ].map((item, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                    <span>{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Model Performance Table */}
          <div className="bg-white rounded-2xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">جدول عملکرد مدل‌ها</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-right py-3 px-4 font-semibold">نام مدل</th>
                    <th className="text-right py-3 px-4 font-semibold">دقت</th>
                    <th className="text-right py-3 px-4 font-semibold">F1-Score</th>
                    <th className="text-right py-3 px-4 font-semibold">Precision</th>
                    <th className="text-right py-3 px-4 font-semibold">Recall</th>
                    <th className="text-right py-3 px-4 font-semibold">وضعیت</th>
                  </tr>
                </thead>
                <tbody>
                  {modelComparisonData.map((model, index) => (
                    <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium">{model.name}</td>
                      <td className="py-3 px-4">{model.accuracy.toFixed(1)}%</td>
                      <td className="py-3 px-4">{model.f1Score.toFixed(1)}%</td>
                      <td className="py-3 px-4">{model.precision.toFixed(1)}%</td>
                      <td className="py-3 px-4">{model.recall.toFixed(1)}%</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          model.status === 'training' ? 'bg-blue-100 text-blue-700' :
                          model.status === 'completed' ? 'bg-green-100 text-green-700' :
                          model.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {model.status === 'training' ? 'در حال آموزش' :
                           model.status === 'completed' ? 'تکمیل شده' :
                           model.status === 'pending' ? 'در انتظار' : 'خطا'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Data Tab */}
      {activeTab === 'data' && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Data Source Performance */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">عملکرد منابع داده</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dataSources}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" stroke="#6b7280" fontSize={10} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="documents" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Data Quality Distribution */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">توزیع کیفیت داده</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RePieChart>
                  <Pie
                    data={[
                      { name: 'عالی (90-100%)', value: 15000, fill: '#10B981' },
                      { name: 'خوب (80-89%)', value: 12000, fill: '#3B82F6' },
                      { name: 'متوسط (70-79%)', value: 5000, fill: '#F59E0B' },
                      { name: 'ضعیف (<70%)', value: 1266, fill: '#EF4444' }
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="value"
                  >
                  </Pie>
                  <Tooltip />
                </RePieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Data Collection Trends */}
          <div className="bg-white rounded-2xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">روند جمع‌آوری داده</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={analyticsData.throughputData}>
                <defs>
                  <linearGradient id="throughputGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                <Area type="monotone" dataKey="processedSamples" stroke="#8B5CF6" fill="url(#throughputGradient)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Resource Utilization */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">استفاده از منابع</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analyticsData.resourceUtilization}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="hour" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Area type="monotone" dataKey="cpuUsage" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="memoryUsage" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="gpuUsage" stackId="1" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Throughput Analysis */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">تحلیل نرخ پردازش</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={analyticsData.throughputData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="batchSize" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                  <Line type="monotone" dataKey="throughput" stroke="#EF4444" strokeWidth={3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Project Performance */}
          <div className="bg-white rounded-2xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">عملکرد پروژه‌ها</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={projectPerformanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" stroke="#6b7280" fontSize={10} />
                <YAxis stroke="#6b7280" fontSize={12} />
                <Tooltip />
                <Bar dataKey="progress" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="accuracy" fill="#10B981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Trends Tab */}
      {activeTab === 'trends' && (
        <>
          <div className="grid grid-cols-1 gap-6">
            {/* Accuracy Trends */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">روند دقت مدل‌ها</h3>
              <ResponsiveContainer width="100%" height={400}>
                <ReLineChart data={analyticsData.accuracyTrends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="precision" stroke="#10B981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="recall" stroke="#F59E0B" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="f1Score" stroke="#8B5CF6" strokeWidth={2} dot={false} />
                </ReLineChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-4 gap-4 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>دقت</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Precision</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span>Recall</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span>F1-Score</span>
                </div>
              </div>
            </div>

            {/* Error Rate Trends */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">روند نرخ خطا</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analyticsData.errorRates}>
                  <defs>
                    <linearGradient id="errorGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip />
                  <Area type="monotone" dataKey="errorRate" stroke="#EF4444" fill="url(#errorGradient)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Full Screen Chart Modal */}
      {fullScreenChart && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-7xl h-5/6 p-6 relative">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">
                {fullScreenChart === 'performance' ? 'روند عملکرد مدل‌ها - نمای کامل' : 
                 fullScreenChart === 'data-quality' ? 'روند کیفیت داده - نمای کامل' : 
                 'نمودار تفصیلی'}
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
                {fullScreenChart === 'performance' ? (
                  <ReLineChart data={analyticsData.modelPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="date" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip />
                    <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" strokeWidth={4} dot={false} />
                    <Line type="monotone" dataKey="f1Score" stroke="#10B981" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="precision" stroke="#F59E0B" strokeWidth={3} dot={false} />
                    <Line type="monotone" dataKey="recall" stroke="#8B5CF6" strokeWidth={3} dot={false} />
                  </ReLineChart>
                ) : (
                  <AreaChart data={analyticsData.dataQualityTrends}>
                    <defs>
                      <linearGradient id="qualityGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="date" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip />
                    <Area type="monotone" dataKey="quality" stroke="#10B981" fill="url(#qualityGradientFull)" strokeWidth={3} />
                    <Area type="monotone" dataKey="consistency" stroke="#3B82F6" fill="transparent" strokeWidth={3} />
                    <Area type="monotone" dataKey="coverage" stroke="#F59E0B" fill="transparent" strokeWidth={3} />
                    <Area type="monotone" dataKey="completeness" stroke="#8B5CF6" fill="transparent" strokeWidth={3} />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyticsPage;