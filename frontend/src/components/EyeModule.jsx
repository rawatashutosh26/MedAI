import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage } from '../api';
import { Upload, Eye, Activity, Layers, ArrowRightLeft, AlertCircle } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import AnimatedCard from './AnimatedCard';

export default function EyeModule({ addToast }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState('report');
  const [dragActive, setDragActive] = useState(false);

  const handleAnalyze = async () => {
    if (!file) {
      addToast('Please select an image first', 'warning');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await analyzeImage('eye', file);
      setResult(data.data);
      setActiveView('report');
      addToast('Analysis completed successfully', 'success');
    } catch (err) {
      addToast('Analysis failed. Please try again.', 'error');
    }
    setLoading(false);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4 mb-8"
      >
        <motion.div
          whileHover={{ rotate: 360, scale: 1.1 }}
          transition={{ duration: 0.6 }}
          className="p-4 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl shadow-lg"
        >
          <Eye className="text-white" size={36} />
        </motion.div>
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Retinal Disease Analysis</h2>
          <p className="text-slate-500">Ensemble Deep Learning with CLAHE Enhancement</p>
        </div>
      </motion.div>
      
      <div className="grid md:grid-cols-3 gap-8">
        {/* LEFT: Upload Section */}
        <div className="md:col-span-1 space-y-6">
          <AnimatedCard delay={0.1}>
            <div className="p-6">
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl h-64 flex flex-col items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 cursor-pointer relative overflow-hidden transition-all duration-300 ${
                  dragActive ? 'border-amber-500 bg-amber-50 scale-105' : 'border-slate-300 hover:border-amber-400'
                }`}
              >
                <input
                  type="file"
                  className="absolute inset-0 opacity-0 cursor-pointer z-10"
                  onChange={(e) => setFile(e.target.files[0])}
                  accept="image/*"
                />
                {file ? (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="relative w-full h-full"
                  >
                    <img
                      src={URL.createObjectURL(file)}
                      className="w-full h-full object-contain rounded-lg"
                      alt="Preview"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                      className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                    >
                      <AlertCircle size={16} />
                    </button>
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center text-slate-400"
                  >
                    <motion.div
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <Upload size={48} className="mx-auto mb-4 text-amber-400" />
                    </motion.div>
                    <p className="text-sm font-medium mb-1">Drop Fundus Image here</p>
                    <p className="text-xs">or click to browse</p>
                  </motion.div>
                )}
              </div>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="w-full mt-4 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 text-white py-3 rounded-lg font-bold transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="sm" text="" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Activity size={20} />
                    <span>Run Diagnostics</span>
                  </>
                )}
              </motion.button>
            </div>
          </AnimatedCard>
          
          {result && (
            <AnimatedCard delay={0.2}>
              <div className="overflow-hidden">
                {['report', 'clahe', 'explain'].map((view, idx) => (
                  <motion.button
                    key={view}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    onClick={() => setActiveView(view)}
                    className={`w-full text-left px-6 py-4 border-b border-slate-200 last:border-b-0 flex items-center gap-3 transition-all ${
                      activeView === view
                        ? 'bg-gradient-to-r from-amber-50 to-orange-50 text-amber-700 font-bold'
                        : 'hover:bg-slate-50 text-slate-600'
                    }`}
                  >
                    {view === 'report' && <Activity size={18} />}
                    {view === 'clahe' && <ArrowRightLeft size={18} />}
                    {view === 'explain' && <Layers size={18} />}
                    <span className="capitalize">
                      {view === 'report' && 'Diagnostic Report'}
                      {view === 'clahe' && 'CLAHE Enhanced View'}
                      {view === 'explain' && 'Saliency Maps'}
                    </span>
                  </motion.button>
                ))}
              </div>
            </AnimatedCard>
          )}
        </div>

        {/* RIGHT: Results Area */}
        <div className="md:col-span-2">
          <AnimatedCard delay={0.2}>
            <div className="p-8 min-h-[600px]">
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key={activeView}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.3 }}
                    className="h-full"
                  >
                    {activeView === 'report' && (
                      <div className="flex flex-col items-center justify-center text-center h-full">
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: 'spring', bounce: 0.4 }}
                          className={`text-8xl mb-6 ${result.diagnosis === 'No DR' ? 'text-green-500' : 'text-amber-500'}`}
                        >
                          {result.diagnosis === 'No DR' ? '✅' : '⚠️'}
                        </motion.div>
                        <motion.h3
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2 }}
                          className="text-4xl font-bold text-slate-800 mb-2"
                        >
                          {result.diagnosis}
                        </motion.h3>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.3 }}
                          className="text-slate-400 font-medium uppercase tracking-widest mb-8"
                        >
                          Clinical Classification
                        </motion.p>
                        
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.4 }}
                          className="w-full max-w-md bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl p-6 border border-amber-200 shadow-sm"
                        >
                          <div className="flex justify-between items-end mb-3">
                            <span className="font-bold text-slate-600">AI Confidence</span>
                            <span className="text-3xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent">
                              {result.confidence}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-4 overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${result.confidence}%` }}
                              transition={{ duration: 1, delay: 0.5 }}
                              className={`h-full rounded-full bg-gradient-to-r ${
                                result.diagnosis === 'No DR'
                                  ? 'from-green-500 to-emerald-500'
                                  : 'from-amber-500 to-orange-500'
                              }`}
                            />
                          </div>
                        </motion.div>
                      </div>
                    )}

                    {activeView === 'clahe' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-amber-700">
                          <ArrowRightLeft className="text-amber-600" size={24} />
                          Standard vs. CLAHE Enhanced
                        </h3>
                        <div className="grid grid-cols-2 gap-4 h-[450px]">
                          <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-slate-200 shadow-md"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.original}`}
                              className="w-full h-full object-cover"
                              alt="Original"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent text-white text-center py-3 text-sm font-bold">
                              Original Scan
                            </div>
                          </motion.div>
                          <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-amber-300 shadow-md"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.clahe}`}
                              className="w-full h-full object-cover"
                              alt="CLAHE"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-amber-600/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              CLAHE Enhanced
                            </div>
                          </motion.div>
                        </div>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mt-4 text-sm text-slate-500 bg-gradient-to-r from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-100"
                        >
                          ℹ️ <b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b> reveals hidden blood vessels and micro-aneurysms that are difficult to see in the original dark scan.
                        </motion.p>
                      </div>
                    )}
                    
                    {activeView === 'explain' && (
                      <div className="h-full flex flex-col items-center">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-red-600">
                          <Layers className="text-red-500" size={24} />
                          AI Attention Map (Grad-CAM)
                        </h3>
                        
                        <div className="grid grid-cols-2 gap-6 w-full max-w-4xl">
                          {/* Legend / Explanation */}
                          <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="flex flex-col justify-center text-left space-y-4"
                          >
                            <div className="bg-gradient-to-br from-slate-50 to-white p-5 rounded-lg border border-slate-200 shadow-sm">
                              <h4 className="font-bold text-slate-700 mb-3">How to read this:</h4>
                              <ul className="text-sm space-y-3 text-slate-600">
                                <li className="flex items-center gap-2">
                                  <span className="w-4 h-4 rounded-full bg-red-600 shadow"></span>
                                  <b>Red Areas:</b> High Importance. The AI found lesions here.
                                </li>
                                <li className="flex items-center gap-2">
                                  <span className="w-4 h-4 rounded-full bg-blue-600 shadow"></span>
                                  <b>Blue Areas:</b> Low Importance. Healthy tissue.
                                </li>
                              </ul>
                            </div>
                            <p className="text-xs text-slate-400">
                              *Generated dynamically using Gradient-weighted Class Activation Mapping on the CNN layer.
                            </p>
                          </motion.div>

                          {/* The Image */}
                          <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-slate-200 shadow-xl"
                          >
                            {result.images.heatmap ? (
                              <img
                                src={`data:image/png;base64,${result.images.heatmap}`}
                                className="w-full h-full object-cover"
                                alt="Grad-CAM Heatmap"
                              />
                            ) : (
                              <div className="h-64 flex items-center justify-center bg-slate-100 text-slate-400">
                                Heatmap unavailable for this model architecture.
                              </div>
                            )}
                          </motion.div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full flex flex-col items-center justify-center text-slate-400"
                  >
                    <motion.div
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <Eye size={64} className="mb-4 opacity-20" />
                    </motion.div>
                    <p className="italic">Select a scan to view detailed analysis.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </AnimatedCard>
        </div>
      </div>
    </div>
  );
}
