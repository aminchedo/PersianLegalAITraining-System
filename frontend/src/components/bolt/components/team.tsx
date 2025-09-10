import React, { useState } from 'react';
import { Users, UserPlus, Mail, Phone, Calendar, Activity, Award, Settings, MoreVertical, Star, TrendingUp, Clock } from 'lucide-react';

const TeamPage: React.FC = () => {
  const [selectedMember, setSelectedMember] = useState<string>('');
  const [showAddModal, setShowAddModal] = useState(false);

  const teamMembers = [
    {
      id: '1',
      name: 'علی احمدی',
      role: 'مدیر سیستم',
      email: 'ali@company.com',
      phone: '09121234567',
      avatar: '',
      status: 'online',
      joinDate: '2023-01-15',
      permissions: ['admin', 'training', 'data'],
      recentActivity: 'راه‌اندازی مدل جدید',
      performance: {
        modelsCreated: 12,
        successRate: 94,
        totalHours: 340,
      }
    },
    {
      id: '2', 
      name: 'فاطمه رضایی',
      role: 'متخصص داده',
      email: 'fateme@company.com',
      phone: '09129876543',
      avatar: '',
      status: 'online',
      joinDate: '2023-03-20',
      permissions: ['data', 'analytics'],
      recentActivity: 'تحلیل کیفیت داده‌های جدید',
      performance: {
        modelsCreated: 8,
        successRate: 91,
        totalHours: 280,
      }
    },
    {
      id: '3',
      name: 'محمد کریمی',
      role: 'مهندس یادگیری ماشین',
      email: 'mohammad@company.com', 
      phone: '09121111111',
      avatar: '',
      status: 'away',
      joinDate: '2023-02-10',
      permissions: ['training', 'models'],
      recentActivity: 'بهینه‌سازی مدل طبقه‌بندی',
      performance: {
        modelsCreated: 15,
        successRate: 89,
        totalHours: 390,
      }
    },
    {
      id: '4',
      name: 'زهرا موسوی',
      role: 'توسعه‌دهنده',
      email: 'zahra@company.com',
      phone: '09122222222', 
      avatar: '',
      status: 'offline',
      joinDate: '2023-04-05',
      permissions: ['api', 'monitoring'],
      recentActivity: 'به‌روزرسانی API',
      performance: {
        modelsCreated: 5,
        successRate: 87,
        totalHours: 220,
      }
    }
  ];

  const roles = [
    { id: 'admin', name: 'مدیر سیستم', color: 'bg-red-100 text-red-700', permissions: ['همه دسترسی‌ها'] },
    { id: 'ml-engineer', name: 'مهندس یادگیری ماشین', color: 'bg-blue-100 text-blue-700', permissions: ['آموزش', 'مدل‌ها'] },
    { id: 'data-scientist', name: 'متخصص داده', color: 'bg-green-100 text-green-700', permissions: ['داده', 'تحلیل'] },
    { id: 'developer', name: 'توسعه‌دهنده', color: 'bg-purple-100 text-purple-700', permissions: ['API', 'نظارت'] },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      default: return 'bg-gray-400';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'online': return 'آنلاین';
      case 'away': return 'غایب';
      default: return 'آفلاین';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">مدیریت تیم</h1>
            <p className="text-pink-100">مدیریت اعضای تیم، نقش‌ها و دسترسی‌ها</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-center">
                <p className="text-2xl font-bold">{teamMembers.length}</p>
                <p className="text-sm text-pink-100">عضو تیم</p>
              </div>
            </div>
            <button 
              onClick={() => setShowAddModal(true)}
              className="bg-white text-pink-600 px-6 py-2 rounded-lg hover:bg-gray-100 transition-colors font-semibold"
            >
              <UserPlus className="w-5 h-5 ml-2 inline" />
              عضو جدید
            </button>
          </div>
        </div>
      </div>

      {/* Team Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-lg ml-4">
              <Users className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">اعضای فعال</p>
              <p className="text-2xl font-bold text-gray-900">
                {teamMembers.filter(m => m.status === 'online').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-lg ml-4">
              <Award className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">میانگین عملکرد</p>
              <p className="text-2xl font-bold text-gray-900">90%</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-purple-100 p-3 rounded-lg ml-4">
              <Activity className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">پروژه‌های فعال</p>
              <p className="text-2xl font-bold text-gray-900">7</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-orange-100 p-3 rounded-lg ml-4">
              <Clock className="w-6 h-6 text-orange-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">ساعات کاری</p>
              <p className="text-2xl font-bold text-gray-900">1.2K</p>
            </div>
          </div>
        </div>
      </div>

      {/* Team Members */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900">اعضای تیم</h2>
              <div className="flex items-center space-x-2 space-x-reverse">
                <button className="text-gray-600 hover:text-gray-900 transition-colors">
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="space-y-4">
              {teamMembers.map((member) => (
                <div key={member.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 space-x-reverse">
                      <div className="relative">
                        <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-blue-600 rounded-full flex items-center justify-center">
                          <span className="text-white font-semibold text-lg">
                            {member.name.split(' ')[0][0]}
                          </span>
                        </div>
                        <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-white ${getStatusColor(member.status)}`}></div>
                      </div>
                      
                      <div>
                        <h3 className="font-semibold text-gray-900">{member.name}</h3>
                        <p className="text-sm text-gray-600">{member.role}</p>
                        <div className="flex items-center space-x-4 space-x-reverse mt-1 text-xs text-gray-500">
                          <span className="flex items-center space-x-1 space-x-reverse">
                            <Mail className="w-3 h-3" />
                            <span>{member.email}</span>
                          </span>
                          <span className="flex items-center space-x-1 space-x-reverse">
                            <Phone className="w-3 h-3" />
                            <span>{member.phone}</span>
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4 space-x-reverse">
                      <div className="text-left">
                        <div className="flex items-center space-x-1 space-x-reverse mb-1">
                          <span className={`w-2 h-2 rounded-full ${getStatusColor(member.status)}`}></span>
                          <span className="text-xs text-gray-600">{getStatusText(member.status)}</span>
                        </div>
                        <p className="text-xs text-gray-500">
                          آخرین فعالیت: {member.recentActivity}
                        </p>
                      </div>
                      
                      <button 
                        onClick={() => setSelectedMember(selectedMember === member.id ? '' : member.id)}
                        className="text-gray-600 hover:text-gray-900 transition-colors"
                      >
                        <MoreVertical className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {selectedMember === member.id && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div className="text-center p-3 bg-blue-50 rounded-lg">
                          <p className="text-lg font-bold text-blue-900">{member.performance.modelsCreated}</p>
                          <p className="text-xs text-blue-700">مدل‌های ایجاد شده</p>
                        </div>
                        <div className="text-center p-3 bg-green-50 rounded-lg">
                          <p className="text-lg font-bold text-green-900">{member.performance.successRate}%</p>
                          <p className="text-xs text-green-700">نرخ موفقیت</p>
                        </div>
                        <div className="text-center p-3 bg-purple-50 rounded-lg">
                          <p className="text-lg font-bold text-purple-900">{member.performance.totalHours}</p>
                          <p className="text-xs text-purple-700">ساعات کاری</p>
                        </div>
                      </div>

                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">دسترسی‌ها</h4>
                        <div className="flex flex-wrap gap-2">
                          {member.permissions.map((permission, index) => (
                            <span 
                              key={index} 
                              className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                            >
                              {permission}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2 space-x-reverse">
                        <button className="bg-blue-600 text-white px-3 py-1 text-sm rounded hover:bg-blue-700 transition-colors">
                          ویرایش پروفایل
                        </button>
                        <button className="border border-gray-300 text-gray-700 px-3 py-1 text-sm rounded hover:bg-gray-50 transition-colors">
                          مدیریت دسترسی
                        </button>
                        <button className="text-red-600 hover:text-red-700 transition-colors px-3 py-1 text-sm">
                          غیرفعال‌سازی
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Roles & Permissions */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">نقش‌های سازمانی</h3>
            
            <div className="space-y-3">
              {roles.map((role) => (
                <div key={role.id} className="border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`px-2 py-1 text-xs rounded-full font-medium ${role.color}`}>
                      {role.name}
                    </span>
                    <span className="text-xs text-gray-600">
                      {teamMembers.filter(m => m.role === role.name).length} نفر
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">
                    دسترسی‌ها: {role.permissions.join('، ')}
                  </div>
                </div>
              ))}
            </div>
            
            <button className="w-full mt-4 bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors">
              مدیریت نقش‌ها
            </button>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">فعالیت‌های اخیر</h3>
            
            <div className="space-y-3">
              <div className="flex items-center space-x-3 space-x-reverse p-2 bg-green-50 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-green-800">علی احمدی مدل جدید راه‌اندازی کرد</p>
                  <p className="text-xs text-green-600">10 دقیقه پیش</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3 space-x-reverse p-2 bg-blue-50 rounded-lg">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-blue-800">فاطمه رضایی داده‌های جدید تحلیل کرد</p>
                  <p className="text-xs text-blue-600">30 دقیقه پیش</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3 space-x-reverse p-2 bg-purple-50 rounded-lg">
                <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-purple-800">محمد کریمی مدل بهینه‌سازی کرد</p>
                  <p className="text-xs text-purple-600">1 ساعت پیش</p>
                </div>
              </div>
            </div>
            
            <button className="w-full mt-4 text-center text-blue-600 hover:text-blue-700 transition-colors text-sm">
              مشاهده همه فعالیت‌ها
            </button>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">عملکرد تیم</h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-600">بهره‌وری</span>
                  <span className="text-sm font-medium text-gray-900">92%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-green-600 h-2 rounded-full" style={{ width: '92%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-600">کیفیت کار</span>
                  <span className="text-sm font-medium text-gray-900">88%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '88%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-600">همکاری تیمی</span>
                  <span className="text-sm font-medium text-gray-900">95%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '95%' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Add Member Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-bold text-gray-900 mb-4">افزودن عضو جدید</h2>
            
            <form className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">نام کامل</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="نام و نام خانوادگی"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">ایمیل</label>
                <input
                  type="email"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="example@company.com"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">نقش</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                  <option>نقش را انتخاب کنید</option>
                  {roles.map((role) => (
                    <option key={role.id} value={role.id}>{role.name}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">شماره تلفن</label>
                <input
                  type="tel"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="09123456789"
                />
              </div>
            </form>
            
            <div className="flex items-center space-x-3 space-x-reverse mt-6">
              <button className="flex-1 bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors">
                افزودن عضو
              </button>
              <button 
                onClick={() => setShowAddModal(false)}
                className="flex-1 border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors"
              >
                انصراف
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamPage;