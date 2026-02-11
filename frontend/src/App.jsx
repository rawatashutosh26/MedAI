import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Brain, 
  Eye, 
  Files, 
  LayoutDashboard, 
  Stethoscope,
  TrendingUp,
  Users,
  Clock,
  Sparkles,
  Menu,
  X,
  Zap,
  Shield,
  Microscope,
  Heart
} from 'lucide-react';
import ChestModule from './components/ChestModule';
import BrainModule from './components/BrainModule';
import EyeModule from './components/EyeModule';
import SkinModule from './components/SkinModule';
import HistoryModule from './components/HistoryModule';
import { ToastContainer } from './components/Toast';

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toasts, setToasts] = useState([]);

  const addToast = (message, type = 'info', duration = 3000) => {
    const id = Date.now();
    setToasts([...toasts, { id, message, type, duration }]);
  };

  const removeToast = (id) => {
    setToasts(toasts.filter(toast => toast.id !== id));
  };

  const renderContent = () => {
    switch(activeTab) {
      case 'home': return <HomeDashboard onNavigate={setActiveTab} addToast={addToast} />;
      case 'chest': return <ChestModule addToast={addToast} />;
      case 'brain': return <BrainModule addToast={addToast} />;
      case 'eye': return <EyeModule addToast={addToast} />;
      case 'skin': return <SkinModule addToast={addToast} />;
      case 'history': return <HistoryModule addToast={addToast} />;
      default: return <div>Select a module</div>;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-purple-50/20 overflow-hidden">
      {/* Sidebar */}
      <motion.aside 
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className={`${sidebarOpen ? 'w-72' : 'w-20'} bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col transition-all duration-300 shadow-2xl relative z-10`}
      >
        {/* Header */}
        <div className="p-6 border-b border-slate-700/50 flex items-center gap-3">
          <motion.div 
            className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg"
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.6 }}
          >
            <Activity className="text-white" size={24} />
          </motion.div>
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex-1"
            >
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                MedAI Pro
              </h1>
              <p className="text-xs text-slate-400">Advanced Diagnostics</p>
            </motion.div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          <SidebarItem 
            icon={<LayoutDashboard size={20} />} 
            label="Dashboard" 
            active={activeTab === 'home'} 
            onClick={() => setActiveTab('home')}
            sidebarOpen={sidebarOpen}
          />
          
          {sidebarOpen && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-xs font-semibold text-slate-500 uppercase mt-6 mb-2 pl-2 tracking-wider"
            >
              Diagnosis Modules
            </motion.div>
          )}
          
          <SidebarItem 
            icon={<Stethoscope size={20} />} 
            label="Chest X-Ray" 
            active={activeTab === 'chest'} 
            onClick={() => setActiveTab('chest')}
            sidebarOpen={sidebarOpen}
            color="blue"
          />
          <SidebarItem 
            icon={<Brain size={20} />} 
            label="Brain MRI" 
            active={activeTab === 'brain'} 
            onClick={() => setActiveTab('brain')}
            sidebarOpen={sidebarOpen}
            color="purple"
          />
          <SidebarItem 
            icon={<Eye size={20} />} 
            label="Retinal Scan" 
            active={activeTab === 'eye'} 
            onClick={() => setActiveTab('eye')}
            sidebarOpen={sidebarOpen}
            color="amber"
          />
          <SidebarItem 
            icon={<Activity size={20} />} 
            label="Skin Lesion" 
            active={activeTab === 'skin'} 
            onClick={() => setActiveTab('skin')}
            sidebarOpen={sidebarOpen}
            color="red"
          />
          
          {sidebarOpen && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-xs font-semibold text-slate-500 uppercase mt-6 mb-2 pl-2 tracking-wider"
            >
              Records
            </motion.div>
          )}
          
          <SidebarItem 
            icon={<Files size={20} />} 
            label="Patient History" 
            active={activeTab === 'history'} 
            onClick={() => setActiveTab('history')}
            sidebarOpen={sidebarOpen}
          />
        </nav>

        {/* Footer */}
        {sidebarOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-4 border-t border-slate-700/50"
          >
            <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 rounded-lg p-3 border border-blue-500/30">
              <div className="flex items-center gap-2 mb-1">
                <Sparkles size={14} className="text-blue-400" />
                <span className="text-xs font-semibold text-slate-300">AI Powered</span>
              </div>
              <p className="text-xs text-slate-400">Advanced ML Models</p>
            </div>
          </motion.div>
        )}
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Toast Container */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}

// Enhanced Home Dashboard
function HomeDashboard({ onNavigate, addToast }) {
  const stats = [
    { 
      label: 'AI-Powered Analysis', 
      value: 'Deep Learning', 
      desc: 'Advanced neural networks for accurate disease detection',
      icon: <Sparkles size={24} />, 
      color: 'blue'
    },
    { 
      label: 'Multi-Modal Detection', 
      value: '4 Modules', 
      desc: 'Brain, Chest, Eye & Skin disease analysis',
      icon: <Microscope size={24} />, 
      color: 'purple'
    },
    { 
      label: 'Fast Processing', 
      value: 'Real-Time', 
      desc: 'Get results in seconds with instant analysis',
      icon: <Zap size={24} />, 
      color: 'amber'
    },
    { 
      label: 'Medical Grade', 
      value: 'FDA Ready', 
      desc: 'Built with clinical accuracy standards',
      icon: <Shield size={24} />, 
      color: 'green'
    },
  ];

  const modules = [
    { 
      id: 'chest',
      title: 'Chest X-Ray', 
      desc: 'Pneumonia & 13 other conditions', 
      icon: <Stethoscope className="text-blue-500" size={32} />,
      gradient: 'from-blue-500 to-cyan-500',
      stats: '14 Conditions',
      onClick: () => onNavigate('chest')
    },
    { 
      id: 'brain',
      title: 'Brain MRI', 
      desc: 'Tumor Detection & Classification', 
      icon: <Brain className="text-purple-500" size={32} />,
      gradient: 'from-purple-500 to-pink-500',
      stats: '4 Classes',
      onClick: () => onNavigate('brain')
    },
    { 
      id: 'eye',
      title: 'Retinal Scan', 
      desc: 'Diabetic Retinopathy Staging', 
      icon: <Eye className="text-amber-500" size={32} />,
      gradient: 'from-amber-500 to-orange-500',
      stats: '5 Stages',
      onClick: () => onNavigate('eye')
    },
    { 
      id: 'skin',
      title: 'Skin Lesion', 
      desc: 'Melanoma Malignancy Check', 
      icon: <Activity className="text-red-500" size={32} />,
      gradient: 'from-red-500 to-rose-500',
      stats: 'Binary Classification',
      onClick: () => onNavigate('skin')
    },
  ];

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-4xl font-bold text-slate-800 mb-2">
          Welcome back, <span className="gradient-text">Dr. User</span> 👋
        </h1>
        <p className="text-slate-600">Here's what's happening with your diagnostics today</p>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, idx) => {
          const colorClasses = {
            blue: { bg: 'bg-blue-100', text: 'text-blue-600', textDark: 'text-blue-700' },
            green: { bg: 'bg-green-100', text: 'text-green-600', textDark: 'text-green-700' },
            purple: { bg: 'bg-purple-100', text: 'text-purple-600', textDark: 'text-purple-700' },
            amber: { bg: 'bg-amber-100', text: 'text-amber-600', textDark: 'text-amber-700' },
          };
          const colors = colorClasses[stat.color] || colorClasses.blue;
          
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white rounded-xl shadow-lg border border-slate-200 p-6 card-hover"
            >
              <div className="flex items-center mb-4">
                <div className={`p-3 ${colors.bg} rounded-lg`}>
                  <div className={colors.text}>
                    {stat.icon}
                  </div>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-slate-800 mb-1">{stat.value}</h3>
              <p className="text-sm font-semibold text-slate-700 mb-2">{stat.label}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{stat.desc}</p>
            </motion.div>
          );
        })}
      </div>

      {/* Modules Grid */}
      <div>
        <motion.h2
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-2xl font-bold text-slate-800 mb-6"
        >
          Diagnostic Modules
        </motion.h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {modules.map((module, idx) => (
            <motion.div
              key={module.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.1 }}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              onClick={module.onClick}
              className="bg-white rounded-xl shadow-lg border border-slate-200 p-6 cursor-pointer card-hover group relative overflow-hidden"
            >
              {/* Gradient Background Effect */}
              <div className={`absolute inset-0 bg-gradient-to-br ${module.gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}></div>
              
              <div className="relative z-10">
                <div className="mb-4 p-4 bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl w-fit group-hover:scale-110 transition-transform duration-300">
                  {module.icon}
                </div>
                <h3 className="font-bold text-xl text-slate-800 mb-2 group-hover:text-slate-900 transition-colors">
                  {module.title}
                </h3>
                <p className="text-slate-500 text-sm mb-3">{module.desc}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold px-3 py-1 rounded-full bg-slate-100 text-slate-600">
                    {module.stats}
                  </span>
                  <motion.div
                    whileHover={{ x: 5 }}
                    className="text-slate-400 group-hover:text-slate-600 transition-colors"
                  >
                    →
                  </motion.div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mt-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-xl p-6 text-white"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold mb-2">Need Help?</h3>
            <p className="text-blue-100">Check our documentation or contact support</p>
          </div>
          <button className="px-6 py-3 bg-white/20 hover:bg-white/30 rounded-lg font-semibold transition-colors backdrop-blur-sm">
            Get Support
          </button>
        </div>
      </motion.div>
    </div>
  );
}

// Enhanced Sidebar Item
function SidebarItem({ icon, label, active, onClick, sidebarOpen, color = 'blue' }) {
  const colorClasses = {
    blue: 'bg-blue-600',
    purple: 'bg-purple-600',
    amber: 'bg-amber-600',
    red: 'bg-red-600',
  };

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ x: 5 }}
      whileTap={{ scale: 0.95 }}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 relative overflow-hidden group ${
        active 
          ? `${colorClasses[color]} text-white shadow-lg` 
          : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
      }`}
    >
      {active && (
        <motion.div
          layoutId="activeTab"
          className={`absolute inset-0 ${colorClasses[color]} rounded-lg`}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
      )}
      <span className="relative z-10">{icon}</span>
      {sidebarOpen && (
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="relative z-10 font-medium"
        >
          {label}
        </motion.span>
      )}
    </motion.button>
  );
}

export default App;
