import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage } from '../api';
import { Upload, Activity, Scissors, Scan, AlertTriangle, CheckCircle, AlertCircle, User, Calendar, MapPin } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import AnimatedCard from './AnimatedCard';

export default function SkinModule({ addToast }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState('report');
  const [dragActive, setDragActive] = useState(false);
  
  // Metadata State
  const [age, setAge] = useState(45);
  const [sex, setSex] = useState("male");
  const [site, setSite] = useState("torso");

  const handleAnalyze = async () => {
    if (!file) {
      addToast('Please select an image first', 'warning');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await analyzeImage('skin', file, { age, sex, site });
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
          className="p-4 bg-gradient-to-br from-red-500 to-rose-500 rounded-xl shadow-lg"
        >
          <Activity className="text-white" size={36} />
        </motion.div>
        <div>
          <h2 className="text-3xl font-bold text-slate-800">Skin Lesion Analysis</h2>
          <p className="text-slate-500">Multi-Modal Analysis (Image + Patient Metadata)</p>
        </div>
      </motion.div>
      
      <div className="grid md:grid-cols-3 gap-8">
        {/* LEFT: Input Section */}
        <div className="md:col-span-1 space-y-6">
          <AnimatedCard delay={0.1}>
            <div className="p-6">
              <h3 className="font-bold text-slate-700 mb-4 border-b pb-3 flex items-center gap-2">
                <User size={18} />
                Patient Metadata
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-slate-500 uppercase font-bold mb-2 block">Age</label>
                  <input
                    type="number"
                    value={age}
                    onChange={e => setAge(e.target.value)}
                    className="w-full p-3 border border-slate-200 rounded-lg bg-slate-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                    min="1"
                    max="120"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-500 uppercase font-bold mb-2 block">Sex</label>
                  <select
                    value={sex}
                    onChange={e => setSex(e.target.value)}
                    className="w-full p-3 border border-slate-200 rounded-lg bg-slate-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-500 uppercase font-bold mb-2 block flex items-center gap-1">
                    <MapPin size={14} />
                    Anatomical Site
                  </label>
                  <select
                    value={site}
                    onChange={e => setSite(e.target.value)}
                    className="w-full p-3 border border-slate-200 rounded-lg bg-slate-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
                  >
                    <option value="torso">Torso (Chest/Back)</option>
                    <option value="head/neck">Head / Neck</option>
                    <option value="upper extremity">Upper Extremity (Arm)</option>
                    <option value="lower extremity">Lower Extremity (Leg)</option>
                    <option value="palms/soles">Palms / Soles</option>
                  </select>
                </div>
              </div>
            </div>
          </AnimatedCard>

          <AnimatedCard delay={0.15}>
            <div className="p-6">
              <h3 className="font-bold text-slate-700 mb-4 border-b pb-3 flex items-center gap-2">
                <Upload size={18} />
                Lesion Image
              </h3>
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-xl h-48 flex flex-col items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 cursor-pointer relative overflow-hidden transition-all duration-300 ${
                  dragActive ? 'border-red-500 bg-red-50 scale-105' : 'border-slate-300 hover:border-red-400'
                }`}
              >
                <input
                  type="file"
                  className="absolute inset-0 opacity-0 cursor-pointer z-10"
                  onChange={e => setFile(e.target.files[0])}
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
                      <Upload size={40} className="mx-auto mb-3 text-red-400" />
                    </motion.div>
                    <p className="text-sm font-medium">Drop image here</p>
                    <p className="text-xs">or click to browse</p>
                  </motion.div>
                )}
              </div>
              {file && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-3 text-xs text-green-600 font-semibold flex items-center gap-1"
                >
                  <CheckCircle size={14} />
                  Image Loaded
                </motion.div>
              )}
            </div>
          </AnimatedCard>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleAnalyze}
            disabled={loading || !file}
            className="w-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 text-white py-4 rounded-lg font-bold shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" text="" />
                <span>Analyzing Lesion...</span>
              </>
            ) : (
              <>
                <Activity size={20} />
                <span>Run Assessment</span>
              </>
            )}
          </motion.button>
          
          {result && (
            <AnimatedCard delay={0.2}>
              <div className="overflow-hidden">
                {['report', 'cleaned', 'segmented'].map((view, idx) => (
                  <motion.button
                    key={view}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    onClick={() => setActiveView(view)}
                    className={`w-full text-left px-6 py-4 border-b border-slate-200 last:border-b-0 flex items-center gap-3 transition-all ${
                      activeView === view
                        ? 'bg-gradient-to-r from-red-50 to-rose-50 text-red-700 font-bold'
                        : 'hover:bg-slate-50 text-slate-600'
                    }`}
                  >
                    {view === 'report' && <Activity size={18} />}
                    {view === 'cleaned' && <Scissors size={18} />}
                    {view === 'segmented' && <Scan size={18} />}
                    <span className="capitalize">
                      {view === 'report' && 'Risk Assessment'}
                      {view === 'cleaned' && 'Hair Removal View'}
                      {view === 'segmented' && 'Lesion Segmentation'}
                    </span>
                  </motion.button>
                ))}
              </div>
            </AnimatedCard>
          )}
        </div>

        {/* RIGHT: Result Section */}
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
                          className={`text-8xl mb-6 ${result.diagnosis === 'Malignant' ? 'text-red-500' : 'text-green-500'}`}
                        >
                          {result.diagnosis === 'Malignant' ? (
                            <AlertTriangle size={96} />
                          ) : (
                            <CheckCircle size={96} />
                          )}
                        </motion.div>
                        
                        <motion.h3
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2 }}
                          className="text-5xl font-bold text-slate-800 mb-2"
                        >
                          {result.diagnosis.toUpperCase()}
                        </motion.h3>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.3 }}
                          className="text-slate-500 font-medium uppercase tracking-widest mb-10"
                        >
                          Based on Metadata & Visual Analysis
                        </motion.p>
                        
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.4 }}
                          className="w-full max-w-lg bg-gradient-to-br from-slate-50 to-white rounded-xl p-8 border border-slate-200 shadow-lg"
                        >
                          <div className="flex justify-between items-end mb-4">
                            <span className="font-bold text-slate-600 text-lg">Malignancy Probability</span>
                            <motion.span
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ delay: 0.5, type: 'spring' }}
                              className={`text-4xl font-mono font-bold ${
                                result.diagnosis === 'Malignant'
                                  ? 'text-red-600'
                                  : 'text-green-600'
                              }`}
                            >
                              {result.malignancy_score}%
                            </motion.span>
                          </div>
                          
                          {/* Custom Progress Bar */}
                          <div className="relative w-full h-8 bg-slate-200 rounded-full overflow-hidden shadow-inner">
                            <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-r from-green-500 via-yellow-400 to-red-600 opacity-30"></div>
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${result.malignancy_score}%` }}
                              transition={{ duration: 1.5, delay: 0.6 }}
                              className={`absolute top-0 left-0 h-full ${
                                result.diagnosis === 'Malignant'
                                  ? 'bg-gradient-to-r from-red-500 to-rose-500'
                                  : 'bg-gradient-to-r from-green-500 to-emerald-500'
                              }`}
                            />
                            <motion.div
                              initial={{ left: 0 }}
                              animate={{ left: `${result.malignancy_score}%` }}
                              transition={{ duration: 1.5, delay: 0.6 }}
                              className="absolute top-0 h-full w-1 bg-slate-800 shadow-lg"
                            />
                          </div>
                          <div className="flex justify-between text-xs text-slate-400 mt-3 font-bold uppercase">
                            <span>Benign</span>
                            <span>Risk Threshold</span>
                            <span>Malignant</span>
                          </div>
                        </motion.div>
                      </div>
                    )}

                    {activeView === 'cleaned' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-red-700">
                          <Scissors className="text-red-600" size={24} />
                          Digital Hair Removal
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
                              Original Lesion
                            </div>
                          </motion.div>
                          <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="relative rounded-lg overflow-hidden border-2 border-red-300 shadow-md"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.cleaned}`}
                              className="w-full h-full object-cover"
                              alt="Cleaned"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-red-600/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              Cleaned (Input to AI)
                            </div>
                          </motion.div>
                        </div>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mt-4 text-sm text-slate-500 bg-gradient-to-r from-red-50 to-rose-50 p-4 rounded-lg border border-red-100"
                        >
                          <b>Why remove hair?</b> Hair strands can confuse the AI model. We use the "DullRazor" algorithm to digitally shave the lesion for accurate classification.
                        </motion.p>
                      </div>
                    )}

                    {activeView === 'segmented' && (
                      <div className="h-full">
                        <h3 className="text-xl font-bold mb-6 flex items-center gap-2 text-red-700">
                          <Scan className="text-red-600" size={24} />
                          Lesion Segmentation
                        </h3>
                        <div className="flex justify-center h-[450px]">
                          <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="relative rounded-lg overflow-hidden border-2 border-slate-200 w-2/3 shadow-xl"
                          >
                            <img
                              src={`data:image/png;base64,${result.images.segmented}`}
                              className="w-full h-full object-cover"
                              alt="Segmented"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-slate-900/90 to-transparent text-white text-center py-3 text-sm font-bold">
                              Boundary Detection
                            </div>
                          </motion.div>
                        </div>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="mt-4 text-sm text-slate-500 bg-gradient-to-r from-red-50 to-rose-50 p-4 rounded-lg border border-red-100 text-center"
                        >
                          <b>ABCD Rule Analysis:</b> The green contour highlights the "Border" of the lesion. Irregular borders are a key sign of melanoma.
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
                      <Activity size={64} className="mb-4 opacity-20" />
                    </motion.div>
                    <p className="italic">Enter patient details and upload image to calculate risk.</p>
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
