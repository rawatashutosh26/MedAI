import React from 'react';
import { motion } from 'framer-motion';

export default function AnimatedCard({ 
  children, 
  className = '', 
  delay = 0,
  hover = true,
  onClick 
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={hover ? { scale: 1.02, y: -4 } : {}}
      className={`bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden ${hover ? 'cursor-pointer transition-shadow duration-300 hover:shadow-2xl' : ''} ${className}`}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
}
