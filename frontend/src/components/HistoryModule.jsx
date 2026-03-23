import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { getHistory } from '../api';
import AnimatedCard from './AnimatedCard';

export default function HistoryModule({ addToast }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('all');

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getHistory();
      setHistory(data);
    } catch {
      addToast('Failed to load history', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

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
          <div className="p-3 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg">
            <i className="bi bi-journal-text text-white" style={{ fontSize: '32px' }} />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-slate-900">Patient History</h2>
            <p className="text-slate-800 font-medium">View and manage diagnostic records</p>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 mb-6">
          <div className="flex-1 relative">
            <i className="bi bi-search text-slate-600 absolute left-3 top-1/2 transform -translate-y-1/2" style={{ fontSize: '20px' }} />
            <input
              type="text"
              placeholder="Search by patient name or module..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-white border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-900"
            />
          </div>
          <div className="relative">
            <i className="bi bi-funnel-fill text-slate-600 absolute left-3 top-1/2 transform -translate-y-1/2" style={{ fontSize: '20px' }} />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="pl-10 pr-8 py-3 bg-white border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none text-slate-900"
            >
              <option value="all">All Modules</option>
              {modules.map(module => (
                <option key={module} value={module}>{module.charAt(0).toUpperCase() + module.slice(1)}</option>
              ))}
            </select>
          </div>
        </div>
      </motion.div>

      {/* Clinical Notice */}
      <div className="mb-8 flex items-start gap-3 rounded-2xl border border-blue-200/60 bg-blue-50 p-4">
        <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-600 text-white shadow">
          <i className="bi bi-shield-lock-fill" style={{ fontSize: '18px' }} />
        </div>
        <div className="text-sm text-slate-900 font-medium leading-relaxed">
          <span className="font-bold text-slate-900">Clinical Notice:</span> History is presented for review; medical decisions should always involve qualified clinicians.
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <AnimatedCard delay={0.1}>
          <div className="p-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-500 text-sm">Total Records</span>
              <i className="bi bi-journal-text text-blue-500" style={{ fontSize: '20px' }} />
            </div>
            <p className="text-3xl font-bold text-slate-800">{history.length}</p>
          </div>
        </AnimatedCard>
        {modules.map((module, idx) => {
          const count = history.filter(h => h.module === module).length;
          const moduleIcon = getModuleIcon(module);
          return (
            <AnimatedCard key={module} delay={0.1 + (idx + 1) * 0.1}>
              <div className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-500 text-sm capitalize">{module}</span>
                  <i className={`${moduleIcon} text-slate-400`} style={{ fontSize: '20px' }} />
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
            <i className="bi bi-journal-text mx-auto mb-4 text-slate-300" style={{ fontSize: '48px' }} />
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
                            <i className={`bi bi-journal-text ${moduleColors.text}`} style={{ fontSize: '24px' }} />
                          </div>
                        );
                      })()}
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-slate-800 mb-1">
                          {item.patientName || 'Anonymous Patient'}
                        </h3>
                        <div className="flex items-center gap-4 text-sm text-slate-500">
                          <span className="flex items-center gap-1">
                            <i className="bi bi-calendar-event" style={{ fontSize: '14px' }} />
                            {new Date(item.timestamp || Date.now()).toLocaleDateString()}
                          </span>
                          <span className="px-2 py-1 bg-slate-100 rounded-full text-xs font-semibold capitalize">
                            {item.module}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                        <i className="bi bi-eye-fill text-slate-400" style={{ fontSize: '20px' }} />
                      </button>
                      <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                        <i className="bi bi-download text-slate-400" style={{ fontSize: '20px' }} />
                      </button>
                      <button className="p-2 hover:bg-red-900 rounded-lg transition-colors">
                        <i className="bi bi-trash-fill text-red-500" style={{ fontSize: '20px' }} />
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

function getModuleIcon(module) {
  const icons = {
    chest: 'bi bi-heart-pulse-fill',
    brain: 'bi bi-brain-fill',
    eye: 'bi bi-eye-fill',
    skin: 'bi bi-person-fill',
  };
  return icons[module] || 'bi bi-journal-text';
}
