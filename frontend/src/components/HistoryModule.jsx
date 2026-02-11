import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Files, Calendar, Search, Filter, Download, Eye, Trash2 } from 'lucide-react';
import { getHistory } from '../api';
import AnimatedCard from './AnimatedCard';

export default function HistoryModule({ addToast }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const data = await getHistory();
      setHistory(data);
    } catch (error) {
      addToast('Failed to load history', 'error');
    } finally {
      setLoading(false);
    }
  };

  const filteredHistory = history.filter(item => {
    const matchesSearch = item.patientName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.module?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filter === 'all' || item.module === filter;
    return matchesSearch && matchesFilter;
  });

  const modules = ['chest', 'brain', 'eye', 'skin'];

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-screen">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full"
        />
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
            <Files className="text-white" size={32} />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-slate-800">Patient History</h2>
            <p className="text-slate-500">View and manage diagnostic records</p>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
            <input
              type="text"
              placeholder="Search by patient name or module..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="pl-10 pr-8 py-3 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none"
            >
              <option value="all">All Modules</option>
              {modules.map(module => (
                <option key={module} value={module}>{module.charAt(0).toUpperCase() + module.slice(1)}</option>
              ))}
            </select>
          </div>
        </div>
      </motion.div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <AnimatedCard delay={0.1}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-500 text-sm">Total Records</span>
              <Files className="text-blue-500" size={20} />
            </div>
            <p className="text-3xl font-bold text-slate-800">{history.length}</p>
          </div>
        </AnimatedCard>
        {modules.map((module, idx) => {
          const count = history.filter(h => h.module === module).length;
          return (
            <AnimatedCard key={module} delay={0.1 + (idx + 1) * 0.1}>
              <div className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-500 text-sm capitalize">{module}</span>
                </div>
                <p className="text-3xl font-bold text-slate-800">{count}</p>
              </div>
            </AnimatedCard>
          );
        })}
      </div>

      {/* History List */}
      {filteredHistory.length === 0 ? (
        <AnimatedCard>
          <div className="p-12 text-center">
            <Files className="mx-auto mb-4 text-slate-300" size={48} />
            <p className="text-slate-500 text-lg">No records found</p>
            <p className="text-slate-400 text-sm mt-2">
              {searchTerm ? 'Try adjusting your search terms' : 'Start analyzing images to see history here'}
            </p>
          </div>
        </AnimatedCard>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item, idx) => (
            <motion.div
              key={item.id || idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
            >
              <AnimatedCard>
                <div className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      {(() => {
                        const moduleColors = getModuleColorClasses(item.module);
                        return (
                          <div className={`p-3 rounded-lg ${moduleColors.bg}`}>
                            <Files className={moduleColors.text} size={24} />
                          </div>
                        );
                      })()}
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-slate-800 mb-1">
                          {item.patientName || 'Anonymous Patient'}
                        </h3>
                        <div className="flex items-center gap-4 text-sm text-slate-500">
                          <span className="flex items-center gap-1">
                            <Calendar size={14} />
                            {new Date(item.timestamp || Date.now()).toLocaleDateString()}
                          </span>
                          <span className="px-2 py-1 bg-slate-100 rounded-full text-xs font-semibold capitalize">
                            {item.module}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
                        <Eye className="text-slate-600" size={20} />
                      </button>
                      <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
                        <Download className="text-slate-600" size={20} />
                      </button>
                      <button className="p-2 hover:bg-red-100 rounded-lg transition-colors">
                        <Trash2 className="text-red-600" size={20} />
                      </button>
                    </div>
                  </div>
                  {item.diagnosis && (
                    <div className="mt-4 pt-4 border-t border-slate-200">
                      <p className="text-sm text-slate-600">
                        <span className="font-semibold">Diagnosis:</span> {item.diagnosis}
                      </p>
                      {item.confidence && (
                        <p className="text-sm text-slate-600 mt-1">
                          <span className="font-semibold">Confidence:</span> {item.confidence}%
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </AnimatedCard>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

function getModuleColorClasses(module) {
  const colors = {
    chest: { bg: 'bg-blue-100', text: 'text-blue-600' },
    brain: { bg: 'bg-purple-100', text: 'text-purple-600' },
    eye: { bg: 'bg-amber-100', text: 'text-amber-600' },
    skin: { bg: 'bg-red-100', text: 'text-red-600' },
  };
  return colors[module] || { bg: 'bg-slate-100', text: 'text-slate-600' };
}
