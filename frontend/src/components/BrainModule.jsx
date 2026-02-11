import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage } from '../api';
import { Upload, Brain, Activity, Layers, Scan, Info, Sparkles, AlertCircle } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import AnimatedCard from './AnimatedCard';

export default function BrainModule({ addToast }) {
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
      const data = await analyzeImage('brain', file);
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

  const getTumorInfo = (type) => {
    const info = {
      'Glioma': "Gliomas occur in the sticky supportive cells (glial cells) that surround nerve cells and help them function. They are the most common type of primary brain tumor.",
      'Meningioma': "A meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. They are often slow-growing.",
      'Pituitary': "Pituitary tumors are abnormal growths that develop in your pituitary gland. Some result in too many hormones causing other problems.",
      'No Tumor': "The MRI scan appears clear. No significant abnormalities were detected by the AI model."
    };
    return info[type] || "Detailed analysis required.";
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
          className="p-4 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg"
        >
          <Brain className="text-white" size={36} />
        </motion.div>
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Brain Tumor Segmentation</h2>
          <p className="text-slate-500">Deep Learning Analysis (VGG-16 Architecture)</p>
        </div>
      </motion.div>
      
      <div className="grid md:grid-cols-3 gap-8">
        {/* LEFT: Controls */}
        <div className="md:col-span-1 space-y-6">
          <AnimatedCard delay={0.1}>
            <div className="p-6">
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl h-64 flex flex-col items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 cursor-pointer relative overflow-hidden transition-all duration-300 ${
                  dragActive ? 'border-purple-500 bg-purple-50 scale-105' : 'border-slate-300 hover:border-purple-400'
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
                      <Upload size={48} className="mx-auto mb-4 text-purple-400" />
                    </motion.div>
                    <p className="text-sm font-medium mb-1">Drop MRI scan here</p>
                    <p className="text-xs">or click to browse</p>
                  </motion.div>
                )}
              </div>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAnalyze}
                disabled={loading || !file}
                className={`w-full mt-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-3 rounded-lg font-bold transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="sm" text="" />
                    <span>Scanning Brain...</span>
                  </>
                ) : (
                  <>
                    <Scan size={20} />
                    <span>Detect Tumor</span>
                  </>
                )}
              </motion.button>
            </div>
          </AnimatedCard>
          
          {result && (
            <AnimatedCard delay={0.2}>
              <div className="overflow-hidden">
                {['report', 'contrast', 'heatmap'].map((view, idx) => (
                  <motion.button
                    key={view}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    onClick={() => setActiveView(view)}
                    className={`w-full text-left px-6 py-4 border-b border-slate-200 last:border-b-0 flex items-center gap-3 transition-all ${
                      activeView === view
                        ? 'bg-gradient-to-r from-purple-50 to-pink-50 text-purple-700 font-bold'
                        : 'hover:bg-slate-50 text-slate-600'
                    }`}
                  >
                    {view === 'report' && <Activity size={18} />}
                    {view === 'contrast' && <Scan size={18} />}
                    {view === 'heatmap' && <Layers size={18} />}
                    <span className="capitalize">
                      {view === 'report' && 'Diagnostic Report'}
                      {view === 'contrast' && 'Contrast Enhanced'}
                      {view === 'heatmap' && 'Tumor Localization'}
                    </span>
                  </motion.button>
                ))}
              </div>
            </AnimatedCard>
          )}
        </div>

        {/* RIGHT: Results */}
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
                          className={`text-8xl mb-6 ${result.diagnosis === 'No Tumor' ? 'text-green-500' : 'text-purple-600'}`}
                        >
                          {result.diagnosis === 'No Tumor' ? '🧠' : '⚠️'}
                        </motion.div>
                        <motion.h3
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2 }}
                          className="text-4xl font-bold text-slate-800 mb-2"
                        >
                          {result.diagnosis}
                        </motion.h3>
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.3 }}
                          className="inline-block bg-gradient-to-r from-purple-100 to-pink-100 px-6 py-2 rounded-full text-sm font-semibold text-purple-700 mb-8"
                        >
                          Confidence: {result.confidence}%
                        </motion.div>
                        
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.4 }}
                          className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 p-6 rounded-xl max-w-lg text-left shadow-sm"
                        >
                          <h4 className="flex items-center gap-2 font-bold text-purple-800 mb-3">
                            <Info size={20} />
                            Medical Context
                          </h4>
                          <p className="text-slate-700 text-sm leading-relaxed">
                            {getTumorInfo(result.diagnosis)}
                          </p>
                        </motion.div>
                      </div>
                    )}

                    {activeView === 'contrast' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-purple-700">
                          <Scan className="text-purple-600" size={24} />
                          Enhanced Visualization
                        </h3>
                        <div className="grid grid-cols-2 gap-4 h-[450px]">
                          <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-slate-200 shadow-md group"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.original}`}
                              className="w-full h-full object-cover"
                              alt="Original"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent text-white text-center py-3 text-sm font-bold">
                              Standard MRI
                            </div>
                          </motion.div>
                          <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-purple-300 shadow-md group"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.contrast}`}
                              className="w-full h-full object-cover"
                              alt="Contrast"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-purple-600/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              High Contrast Mode
                            </div>
                          </motion.div>
                        </div>
                        <p className="mt-4 text-xs text-slate-400 text-center">
                          *Histogram Equalization applied to improve tumor visibility.
                        </p>
                      </div>
                    )}

                    {activeView === 'heatmap' && (
                      <div className="h-full flex flex-col items-center">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-red-600">
                          <Layers className="text-red-500" size={24} />
                          Tumor Heatmap
                        </h3>
                        <motion.div
                          initial={{ scale: 0.9, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          className="relative rounded-lg overflow-hidden border-2 border-slate-200 shadow-xl w-full max-w-md aspect-square"
                        >
                          {result.images.heatmap ? (
                            <img
                              src={`data:image/png;base64,${result.images.heatmap}`}
                              className="w-full h-full object-cover"
                              alt="Heatmap"
                            />
                          ) : (
                            <div className="h-full flex items-center justify-center bg-slate-100 text-slate-400">
                              Heatmap Unavailable
                            </div>
                          )}
                        </motion.div>
                        <div className="mt-6 flex gap-8">
                          <div className="flex items-center gap-2 text-sm text-slate-600">
                            <div className="w-4 h-4 bg-red-600 rounded shadow"></div>
                            Tumor Region
                          </div>
                          <div className="flex items-center gap-2 text-sm text-slate-600">
                            <div className="w-4 h-4 bg-blue-600 rounded shadow"></div>
                            Healthy Tissue
                          </div>
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
                      <Brain size={64} className="mb-4 opacity-20" />
                    </motion.div>
                    <p className="italic">Select an MRI scan to begin segmentation.</p>
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
