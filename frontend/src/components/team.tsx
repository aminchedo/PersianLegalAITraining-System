import React, { useState } from 'react';
import { 
  Users, Plus, X, Edit3, Trash2, Search, Filter, Settings,
  Mail, Phone, Calendar, MapPin, Shield, Crown, User,
  Clock, Activity, CheckCircle, AlertCircle, Eye, EyeOff,
  UserPlus, UserMinus, Key, Award, Target, BarChart3
} from 'lucide-react';
import { useAppContext } from './router';
import { TeamMember, NewMember } from '../types/dashboard';

const TeamPage = () => {
  const { teamMembers } = useAppContext();
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [viewMode, setViewMode] = useState('grid');
  const [showPermissions, setShowPermissions] = useState(false);

  const [newMember, setNewMember] = useState<NewMember>({
    name: '',
    email: '',
    role: 'data-scientist',
    phone: '',
    permissions: [],
    projects: []
  });

  // Extended team data with more realistic information
  const [allMembers, setAllMembers] = useState([
    {
      id: 1,
      name: "علی احمدی",
      role: "مدیر پروژه",
      email: "ali.ahmadi@company.com",
      phone: "+98 912 345 6789",
      avatar: "AA",
      status: "online",
      lastActive: new Date(Date.now() - 5 * 60000),
      joinDate: new Date(2023, 8, 15),
      projects: ["legal-2025", "criminal-ai"],
      permissions: ["admin", "model-training", "data-access", "user-management"],
      department: "هوش مصنوعی",
      location: "تهران، ایران",
      totalTasks: 47,
      completedTasks: 42,
      activeProjects: 2,
      experienceYears: 8,
      skills: ["Project Management", "Machine Learning", "Team Leadership"]
    },
    {
      id: 2,
      name: "فاطمه کریمی",
      role: "مهندس یادگیری ماشین",
      email: "fateme.karimi@company.com",
      phone: "+98 915 123 4567",
      avatar: "FK",
      status: "online",
      lastActive: new Date(Date.now() - 2 * 60000),
      joinDate: new Date(2023, 6, 10),
      projects: ["legal-2025", "civil-qa"],
      permissions: ["model-training", "data-access", "analytics"],
      department: "تحقیق و توسعه",
      location: "تهران، ایران",
      totalTasks: 35,
      completedTasks: 31,
      activeProjects: 2,
      experienceYears: 5,
      skills: ["PyTorch", "Transformers", "NLP", "Persian Language Processing"]
    },
    {
      id: 3,
      name: "محمد رضایی",
      role: "متخصص داده",
      email: "mohammad.rezaei@company.com",
      phone: "+98 916 789 0123",
      avatar: "MR",
      status: "away",
      lastActive: new Date(Date.now() - 15 * 60000),
      joinDate: new Date(2024, 0, 20),
      projects: ["civil-qa"],
      permissions: ["data-access", "data-annotation", "quality-control"],
      department: "مهندسی داده",
      location: "اصفهان، ایران",
      totalTasks: 28,
      completedTasks: 24,
      activeProjects: 1,
      experienceYears: 3,
      skills: ["Data Mining", "Python", "SQL", "Data Visualization"]
    },
    {
      id: 4,
      name: "زهرا موسوی",
      role: "طراح UI/UX",
      email: "zahra.mousavi@company.com",
      phone: "+98 917 456 7890",
      avatar: "ZM",
      status: "offline",
      lastActive: new Date(Date.now() - 2 * 60 * 60000),
      joinDate: new Date(2023, 10, 5),
      projects: ["legal-2025"],
      permissions: ["design-access", "user-testing"],
      department: "طراحی",
      location: "شیراز، ایران",
      totalTasks: 22,
      completedTasks: 20,
      activeProjects: 1,
      experienceYears: 4,
      skills: ["Figma", "User Research", "Prototyping", "Design Systems"]
    },
    {
      id: 5,
      name: "امیرحسین نوری",
      role: "توسعه‌دهنده بک‌اند",
      email: "amirhossein.nouri@company.com",
      phone: "+98 918 234 5678",
      avatar: "AN",
      status: "busy",
      lastActive: new Date(Date.now() - 1 * 60000),
      joinDate: new Date(2023, 7, 1),
      projects: ["legal-2025", "criminal-ai", "civil-qa"],
      permissions: ["backend-development", "api-access", "deployment"],
      department: "توسعه نرم‌افزار",
      location: "تهران، ایران",
      totalTasks: 56,
      completedTasks: 48,
      activeProjects: 3,
      experienceYears: 6,
      skills: ["Django", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"]
    },
    {
      id: 6,
      name: "سارا حسینی",
      role: "مدیر کیفیت",
      email: "sara.hosseini@company.com",
      phone: "+98 919 876 5432",
      avatar: "SH",
      status: "online",
      lastActive: new Date(Date.now() - 8 * 60000),
      joinDate: new Date(2023, 9, 12),
      projects: ["legal-2025", "criminal-ai"],
      permissions: ["quality-control", "testing", "documentation"],
      department: "کیفیت و تست",
      location: "تهران، ایران",
      totalTasks: 31,
      completedTasks: 29,
      activeProjects: 2,
      experienceYears: 7,
      skills: ["Quality Assurance", "Test Automation", "Documentation", "Process Improvement"]
    }
  ]);

  // Available permissions
  const allPermissions = [
    { id: 'admin', label: 'مدیریت کل', category: 'management' },
    { id: 'user-management', label: 'مدیریت کاربران', category: 'management' },
    { id: 'model-training', label: 'آموزش مدل', category: 'ml' },
    { id: 'data-access', label: 'دسترسی داده', category: 'data' },
    { id: 'data-annotation', label: 'برچسب‌گذاری داده', category: 'data' },
    { id: 'quality-control', label: 'کنترل کیفیت', category: 'quality' },
    { id: 'analytics', label: 'آنالیز و گزارش', category: 'analytics' },
    { id: 'deployment', label: 'استقرار سیستم', category: 'development' },
    { id: 'api-access', label: 'دسترسی API', category: 'development' },
    { id: 'backend-development', label: 'توسعه بک‌اند', category: 'development' },
    { id: 'design-access', label: 'دسترسی طراحی', category: 'design' },
    { id: 'user-testing', label: 'تست کاربری', category: 'design' },
    { id: 'testing', label: 'تست نرم‌افزار', category: 'quality' },
    { id: 'documentation', label: 'مستندسازی', category: 'quality' }
  ];

  // Available roles
  const roles = [
    { id: 'admin', label: 'مدیر سیستم', color: 'red' },
    { id: 'project-manager', label: 'مدیر پروژه', color: 'purple' },
    { id: 'ml-engineer', label: 'مهندس یادگیری ماشین', color: 'blue' },
    { id: 'data-scientist', label: 'دانشمند داده', color: 'green' },
    { id: 'data-engineer', label: 'مهندس داده', color: 'indigo' },
    { id: 'backend-developer', label: 'توسعه‌دهنده بک‌اند', color: 'gray' },
    { id: 'ui-designer', label: 'طراح رابط کاربری', color: 'pink' },
    { id: 'qa-engineer', label: 'مهندس کیفیت', color: 'yellow' }
  ];

  // Filter members
  const filteredMembers = allMembers.filter(member => {
    const matchesSearch = member.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         member.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         member.department.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = roleFilter === 'all' || member.role.includes(roleFilter);
    const matchesStatus = statusFilter === 'all' || member.status === statusFilter;
    return matchesSearch && matchesRole && matchesStatus;
  });

  // Get status configuration
  const getStatusConfig = (status: string) => {
    const configs = {
      online: { color: 'text-green-700', bgColor: 'bg-green-100', dot: 'bg-green-500', label: 'آنلاین' },
      away: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', dot: 'bg-yellow-500', label: 'غایب' },
      busy: { color: 'text-red-700', bgColor: 'bg-red-100', dot: 'bg-red-500', label: 'مشغول' },
      offline: { color: 'text-gray-700', bgColor: 'bg-gray-100', dot: 'bg-gray-500', label: 'آفلاین' }
    };
    return configs[status] || configs.offline;
  };

  // Get role color
  const getRoleColor = (role: string) => {
    const roleObj = roles.find(r => role.includes(r.label.split(' ')[0].toLowerCase()));
    return roleObj?.color || 'gray';
  };

  // Handle add member
  const handleAddMember = () => {
    const member = {
      id: Date.now(),
      ...newMember,
      avatar: newMember.name.split(' ').map(n => n[0]).join('').toUpperCase(),
      status: 'offline',
      lastActive: new Date(),
      joinDate: new Date(),
      totalTasks: 0,
      completedTasks: 0,
      activeProjects: newMember.projects.length,
      experienceYears: 0,
      skills: [],
      department: 'جدید',
      location: 'نامشخص'
    };
    
    setAllMembers(prev => [...prev, member]);
    setNewMember({
      name: '',
      email: '',
      role: 'data-scientist',
      phone: '',
      permissions: [],
      projects: []
    });
    setShowAddModal(false);
  };

  // Handle remove member
  const handleRemoveMember = (memberId: any) => {
    setAllMembers(prev => prev.filter(member => member.id !== memberId));
  };

  // Team statistics
  const teamStats = {
    total: allMembers.length,
    online: allMembers.filter(m => m.status === 'online').length,
    departments: [...new Set(allMembers.map(m => m.department))].length,
    avgExperience: (allMembers.reduce((sum, m) => sum + m.experienceYears, 0) / allMembers.length).toFixed(1),
    completionRate: ((allMembers.reduce((sum, m) => sum + m.completedTasks, 0) / 
                     allMembers.reduce((sum, m) => sum + m.totalTasks, 0)) * 100).toFixed(1)
  };

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 via-pink-600 to-red-600 rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Users className="w-8 h-8" />
              مدیریت تیم
            </h1>
            <p className="text-purple-100">مدیریت اعضای تیم، دسترسی‌ها و عملکرد</p>
          </div>
          <button 
            onClick={() => setShowAddModal(true)}
            className="bg-white text-purple-600 px-6 py-3 rounded-xl font-medium hover:bg-purple-50 transition-all duration-300 flex items-center gap-2"
          >
            <UserPlus className="w-5 h-5" />
            عضو جدید
          </button>
        </div>
        
        {/* Team Statistics */}
        <div className="grid grid-cols-5 gap-4 mt-6">
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4" />
              <span className="text-sm">کل اعضا</span>
            </div>
            <p className="text-2xl font-bold">{teamStats.total}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4" />
              <span className="text-sm">آنلاین</span>
            </div>
            <p className="text-2xl font-bold">{teamStats.online}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4" />
              <span className="text-sm">بخش‌ها</span>
            </div>
            <p className="text-2xl font-bold">{teamStats.departments}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Award className="w-4 h-4" />
              <span className="text-sm">میانگین تجربه</span>
            </div>
            <p className="text-2xl font-bold">{teamStats.avgExperience} سال</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4" />
              <span className="text-sm">نرخ تکمیل</span>
            </div>
            <p className="text-2xl font-bold">{teamStats.completionRate}%</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="w-5 h-5 text-gray-400 absolute right-3 top-1/2 transform -translate-y-1/2" />
              <input 
                type="text"
                placeholder="جستجو در اعضای تیم..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-64 pl-4 pr-10 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
            <select 
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">همه نقش‌ها</option>
              <option value="مدیر">مدیر</option>
              <option value="مهندس">مهندس</option>
              <option value="متخصص">متخصص</option>
              <option value="طراح">طراح</option>
              <option value="توسعه‌دهنده">توسعه‌دهنده</option>
            </select>
            <select 
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">همه وضعیت‌ها</option>
              <option value="online">آنلاین</option>
              <option value="away">غایب</option>
              <option value="busy">مشغول</option>
              <option value="offline">آفلاین</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-lg transition-all ${viewMode === 'grid' ? 'bg-purple-100 text-purple-600' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              <Users className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-lg transition-all ${viewMode === 'list' ? 'bg-purple-100 text-purple-600' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setShowPermissions(!showPermissions)}
              className={`p-2 rounded-lg transition-all ${showPermissions ? 'bg-purple-100 text-purple-600' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              <Shield className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Team Members Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMembers.map(member => {
              const statusConfig = getStatusConfig(member.status);
              const roleColor = getRoleColor(member.role);
              
              return (
                <div key={member.id} className="group relative">
                  <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl p-6 border border-gray-200 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <div className={`w-12 h-12 rounded-xl bg-${roleColor}-100 flex items-center justify-center`}>
                            <span className={`text-${roleColor}-600 font-bold`}>{member.avatar}</span>
                          </div>
                          <div className={`absolute -bottom-1 -right-1 w-4 h-4 ${statusConfig.dot} rounded-full border-2 border-white`}></div>
                        </div>
                        <div>
                          <h3 className="font-bold text-gray-900">{member.name}</h3>
                          <p className="text-sm text-gray-600">{member.role}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button 
                          onClick={() => setSelectedMember(member)}
                          className="w-8 h-8 bg-purple-600 text-white rounded-lg flex items-center justify-center hover:bg-purple-700 transition-all"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => handleRemoveMember(member.id)}
                          className="w-8 h-8 bg-red-600 text-white rounded-lg flex items-center justify-center hover:bg-red-700 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Content */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Mail className="w-4 h-4" />
                        <span>{member.email}</span>
                      </div>
                      
                      {member.phone && (
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                          <Phone className="w-4 h-4" />
                          <span>{member.phone}</span>
                        </div>
                      )}
                      
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <MapPin className="w-4 h-4" />
                        <span>{member.location}</span>
                      </div>

                      {/* Stats */}
                      <div className="grid grid-cols-3 gap-3 mt-4 pt-3 border-t border-gray-200">
                        <div className="text-center">
                          <p className="text-lg font-bold text-gray-900">{member.activeProjects}</p>
                          <p className="text-xs text-gray-500">پروژه فعال</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold text-gray-900">{member.completedTasks}</p>
                          <p className="text-xs text-gray-500">وظیفه انجام شده</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold text-gray-900">{member.experienceYears}</p>
                          <p className="text-xs text-gray-500">سال تجربه</p>
                        </div>
                      </div>

                      {/* Status and Last Active */}
                      <div className="flex items-center justify-between mt-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
                          {statusConfig.label}
                        </span>
                        <span className="text-xs text-gray-500">
                          {member.status === 'online' ? 'آنلاین' : 
                           `${Math.floor((new Date() - member.lastActive) / 60000)} دقیقه پیش`}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredMembers.map(member => {
              const statusConfig = getStatusConfig(member.status);
              const roleColor = getRoleColor(member.role);
              
              return (
                <div key={member.id} className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-all">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="relative">
                        <div className={`w-12 h-12 rounded-xl bg-${roleColor}-100 flex items-center justify-center`}>
                          <span className={`text-${roleColor}-600 font-bold`}>{member.avatar}</span>
                        </div>
                        <div className={`absolute -bottom-1 -right-1 w-4 h-4 ${statusConfig.dot} rounded-full border-2 border-white`}></div>
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900">{member.name}</h3>
                        <p className="text-sm text-gray-600">{member.role} • {member.department}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500 mt-1">
                          <span>{member.email}</span>
                          <span>•</span>
                          <span>{member.activeProjects} پروژه فعال</span>
                          <span>•</span>
                          <span>{member.experienceYears} سال تجربه</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="w-32 bg-gray-200 rounded-full h-2 mb-1">
                          <div 
                            className="bg-purple-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${(member.completedTasks / member.totalTasks) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-500">
                          {member.completedTasks}/{member.totalTasks} وظیفه
                        </span>
                      </div>
                      
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
                        {statusConfig.label}
                      </span>
                      
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => setSelectedMember(member)}
                          className="w-8 h-8 bg-purple-600 text-white rounded-lg flex items-center justify-center hover:bg-purple-700 transition-all"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                          <Edit3 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Permissions Panel */}
      {showPermissions && (
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
            <Shield className="w-5 h-5" />
            مدیریت دسترسی‌ها
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(
              allPermissions.reduce((acc, perm) => {
                if (!acc[perm.category]) acc[perm.category] = [];
                acc[perm.category].push(perm);
                return acc;
              }, {})
            ).map(([category, permissions]) => (
              <div key={category} className="bg-gray-50 rounded-xl p-4">
                <h4 className="font-semibold text-gray-900 mb-3 capitalize">
                  {category === 'management' ? 'مدیریت' :
                   category === 'ml' ? 'یادگیری ماشین' :
                   category === 'data' ? 'داده' :
                   category === 'analytics' ? 'آنالیز' :
                   category === 'development' ? 'توسعه' :
                   category === 'design' ? 'طراحی' :
                   category === 'quality' ? 'کیفیت' : category}
                </h4>
                <div className="space-y-2">
                  {permissions.map(permission => (
                    <div key={permission.id} className="text-sm text-gray-700">
                      {permission.label}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Add Member Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">افزودن عضو جدید</h3>
              <button 
                onClick={() => setShowAddModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">نام و نام خانوادگی</label>
                  <input 
                    type="text"
                    value={newMember.name}
                    onChange={(e) => setNewMember(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
                    placeholder="نام و نام خانوادگی..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">ایمیل</label>
                  <input 
                    type="email"
                    value={newMember.email}
                    onChange={(e) => setNewMember(prev => ({ ...prev, email: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
                    placeholder="email@company.com"
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">نقش</label>
                  <select 
                    value={newMember.role}
                    onChange={(e) => setNewMember(prev => ({ ...prev, role: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
                  >
                    {roles.map(role => (
                      <option key={role.id} value={role.id}>{role.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">شماره تماس</label>
                  <input 
                    type="tel"
                    value={newMember.phone}
                    onChange={(e) => setNewMember(prev => ({ ...prev, phone: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500"
                    placeholder="+98 912 345 6789"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">دسترسی‌ها</label>
                <div className="grid grid-cols-3 gap-2 max-h-40 overflow-y-auto p-3 border border-gray-300 rounded-xl">
                  {allPermissions.map(permission => (
                    <label key={permission.id} className="flex items-center gap-2 text-sm">
                      <input 
                        type="checkbox"
                        checked={newMember.permissions.includes(permission.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setNewMember(prev => ({ 
                              ...prev, 
                              permissions: [...prev.permissions, permission.id] 
                            }));
                          } else {
                            setNewMember(prev => ({ 
                              ...prev, 
                              permissions: prev.permissions.filter(p => p !== permission.id) 
                            }));
                          }
                        }}
                        className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                      />
                      <span>{permission.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button 
                onClick={handleAddMember}
                className="flex-1 bg-purple-600 text-white py-3 rounded-xl hover:bg-purple-700 transition-all font-medium"
              >
                افزودن عضو
              </button>
              <button 
                onClick={() => setShowAddModal(false)}
                className="flex-1 bg-gray-300 text-gray-700 py-3 rounded-xl hover:bg-gray-400 transition-all font-medium"
              >
                لغو
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Member Detail Modal */}
      {selectedMember && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-auto p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">جزئیات عضو: {selectedMember.name}</h3>
              <button 
                onClick={() => setSelectedMember(null)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Personal Info */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">اطلاعات شخصی</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>نام:</span>
                      <span>{selectedMember.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>نقش:</span>
                      <span>{selectedMember.role}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>بخش:</span>
                      <span>{selectedMember.department}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>محل:</span>
                      <span>{selectedMember.location}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>تاریخ عضویت:</span>
                      <span>{selectedMember.joinDate.toLocaleDateString('fa-IR')}</span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">تماس</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      <span>{selectedMember.email}</span>
                    </div>
                    {selectedMember.phone && (
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4" />
                        <span>{selectedMember.phone}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Performance Stats */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">آمار عملکرد</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">{selectedMember.activeProjects}</p>
                      <p className="text-xs text-gray-500">پروژه فعال</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">{selectedMember.completedTasks}</p>
                      <p className="text-xs text-gray-500">وظیفه تکمیل شده</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">{selectedMember.totalTasks}</p>
                      <p className="text-xs text-gray-500">کل وظایف</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-600">{selectedMember.experienceYears}</p>
                      <p className="text-xs text-gray-500">سال تجربه</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">مهارت‌ها</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedMember.skills.map((skill, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Projects and Permissions */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">پروژه‌ها</h4>
                  <div className="space-y-2">
                    {selectedMember.projects.map((projectId, index) => (
                      <div key={index} className="text-sm text-gray-700 bg-white p-2 rounded">
                        {projectId}
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">دسترسی‌ها</h4>
                  <div className="space-y-1">
                    {selectedMember.permissions.map((permId, index) => {
                      const perm = allPermissions.find(p => p.id === permId);
                      return perm ? (
                        <div key={index} className="text-sm text-gray-700 bg-white p-2 rounded flex items-center gap-2">
                          <Shield className="w-3 h-3" />
                          {perm.label}
                        </div>
                      ) : null;
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamPage;