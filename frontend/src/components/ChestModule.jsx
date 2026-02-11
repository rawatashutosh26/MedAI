import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage } from '../api';
import { Upload, Stethoscope, Activity, ScanLine, Contrast, Info, AlertCircle } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import AnimatedCard from './AnimatedCard';

export default function ChestModule({ addToast }) {
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
      const data = await analyzeImage('chest', file);
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

  const getConditionInfo = (condition) => {
    const definitions = {
      'Atelectasis': "Partial collapse of the lung or lobe, often due to blockage of air passages.",
      'Cardiomegaly': "Enlarged heart, which can be a sign of heart failure or other cardiovascular issues.",
      'Effusion': "Fluid buildup between the tissues that line the lungs and the chest (Pleural Effusion).",
      'Infiltration': "A substance denser than air (pus, blood, protein) lingering in the lungs.",
      'Mass': "A lesion seen on chest x-ray greater than 3cm in diameter.",
      'Nodule': "A small growth or lump on the lung (less than 3cm).",
      'Pneumonia': "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
      'Pneumothorax': "A collapsed lung caused by air leaking into the space between the lung and chest wall.",
      'No Finding': "The X-Ray appears clear. No significant abnormalities detected within the model's scope."
    };
    return definitions[condition] || "Clinical correlation is recommended for this finding.";
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
          className="p-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl shadow-lg"
        >
          <Stethoscope className="text-white" size={36} />
        </motion.div>
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Chest X-Ray Diagnostics</h2>
          <p className="text-slate-500">DenseNet-121 / ResNet Analysis (NIH Dataset)</p>
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
                  dragActive ? 'border-blue-500 bg-blue-50 scale-105' : 'border-slate-300 hover:border-blue-400'
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
                      <Upload size={48} className="mx-auto mb-4 text-blue-400" />
                    </motion.div>
                    <p className="text-sm font-medium mb-1">Drop X-Ray here</p>
                    <p className="text-xs">or click to browse</p>
                  </motion.div>
                )}
              </div>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="w-full mt-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white py-3 rounded-lg font-bold transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="sm" text="" />
                    <span>Analyzing Thorax...</span>
                  </>
                ) : (
                  <>
                    <Activity size={20} />
                    <span>Run Analysis</span>
                  </>
                )}
              </motion.button>
            </div>
          </AnimatedCard>
          
          {result && (
            <AnimatedCard delay={0.2}>
              <div className="overflow-hidden">
                {['report', 'clahe', 'negative'].map((view, idx) => (
                  <motion.button
                    key={view}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    onClick={() => setActiveView(view)}
                    className={`w-full text-left px-6 py-4 border-b border-slate-200 last:border-b-0 flex items-center gap-3 transition-all ${
                      activeView === view
                        ? 'bg-gradient-to-r from-blue-50 to-cyan-50 text-blue-700 font-bold'
                        : 'hover:bg-slate-50 text-slate-600'
                    }`}
                  >
                    {view === 'report' && <Activity size={18} />}
                    {view === 'clahe' && <ScanLine size={18} />}
                    {view === 'negative' && <Contrast size={18} />}
                    <span className="capitalize">
                      {view === 'report' && 'Pathology Report'}
                      {view === 'clahe' && 'Bone Enhanced (CLAHE)'}
                      {view === 'negative' && 'Negative Mode'}
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
                      <div className="h-full">
                        <div className="flex items-center justify-between mb-6 border-b pb-4">
                          <h3 className="text-2xl font-bold text-slate-800">Detected Conditions</h3>
                          <motion.span
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="text-sm bg-gradient-to-r from-blue-100 to-cyan-100 text-blue-800 px-4 py-2 rounded-full font-semibold"
                          >
                            {result.findings.length} Finding(s)
                          </motion.span>
                        </div>

                        <div className="space-y-4 max-h-[450px] overflow-y-auto pr-2 custom-scrollbar">
                          {result.findings.map((item, idx) => (
                            <motion.div
                              key={idx}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: idx * 0.1 }}
                              className="bg-gradient-to-br from-slate-50 to-white border border-slate-200 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                            >
                              <div className="flex justify-between items-center mb-3">
                                <h4 className="font-bold text-lg text-slate-700">{item.condition}</h4>
                                <motion.span
                                  initial={{ scale: 0 }}
                                  animate={{ scale: 1 }}
                                  transition={{ delay: idx * 0.1 + 0.2 }}
                                  className={`px-4 py-1.5 rounded-full text-sm font-bold ${
                                    item.condition === 'No Finding'
                                      ? 'bg-gradient-to-r from-green-100 to-emerald-100 text-green-700'
                                      : 'bg-gradient-to-r from-red-100 to-rose-100 text-red-700'
                                  }`}
                                >
                                  {item.confidence}%
                                </motion.span>
                              </div>
                              {/* Progress Bar */}
                              <div className="w-full bg-slate-200 rounded-full h-3 mb-3 overflow-hidden">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${item.confidence}%` }}
                                  transition={{ duration: 1, delay: idx * 0.1 + 0.3 }}
                                  className={`h-full rounded-full ${
                                    item.condition === 'No Finding'
                                      ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                                      : 'bg-gradient-to-r from-red-500 to-rose-500'
                                  }`}
                                />
                              </div>
                              {/* Medical Definition */}
                              <div className="flex gap-2 text-xs text-slate-600 bg-white p-3 rounded-lg border border-slate-100">
                                <Info size={16} className="mt-0.5 shrink-0 text-blue-500" />
                                <p>{getConditionInfo(item.condition)}</p>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}

                    {activeView === 'clahe' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-blue-700">
                          <ScanLine className="text-blue-600" size={24} />
                          Bone Enhanced Visualization
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
                              Standard CXR
                            </div>
                          </motion.div>
                          <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-blue-300 shadow-md"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.clahe}`}
                              className="w-full h-full object-cover"
                              alt="CLAHE"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-blue-600/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              Enhanced (CLAHE)
                            </div>
                          </motion.div>
                        </div>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mt-4 text-sm text-slate-500 bg-gradient-to-r from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-100"
                        >
                          <b>Why use this?</b> This filter equalizes the histogram to bring out details in low-contrast areas (like behind the heart or ribs).
                        </motion.p>
                      </div>
                    )}

                    {activeView === 'negative' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-blue-700">
                          <Contrast className="text-blue-600" size={24} />
                          Negative Mode
                        </h3>
                        <div className="flex justify-center h-[450px]">
                          <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="relative rounded-lg overflow-hidden border-2 border-slate-200 w-2/3 shadow-xl"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.negative}`}
                              className="w-full h-full object-cover"
                              alt="Negative"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-slate-900/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              Inverted Scan
                            </div>
                          </motion.div>
                        </div>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mt-4 text-sm text-slate-500 bg-gradient-to-r from-blue-50 to-cyan-50 p-4 rounded-lg border border-blue-100 text-center"
                        >
                          <b>Why use this?</b> Inverting the image helps radiologists spot small nodules or masses, which appear as bright spots against a dark lung background.
                        </motion.p>
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
                      <Stethoscope size={64} className="mb-4 opacity-20" />
                    </motion.div>
                    <p className="italic">Select a Chest X-Ray to detect pathology.</p>
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
