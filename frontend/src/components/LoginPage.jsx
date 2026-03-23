import React, { useState } from 'react';
import { motion } from 'framer-motion';
import LoadingSpinner from './LoadingSpinner';
import { login } from '../api';

export default function LoginPage({ addToast, onSuccess, onSwitchToSignup }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = email.trim().includes('@') && password.length >= 1;

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit || submitting) return;
    setSubmitting(true);
    try {
      await login(email, password);
      addToast('Logged in successfully', 'success');
      onSuccess();
    } catch {
      addToast('Login failed. Check your email/password.', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 max-w-lg mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white border border-blue-100 rounded-3xl shadow-lg overflow-hidden"
      >
        <div className="p-6 border-b border-blue-100/60 bg-gradient-to-r from-blue-50 to-cyan-50">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-600 text-white shadow">
              <i className="bi bi-shield-lock-fill" style={{ fontSize: '20px' }} />
            </div>
            <div>
              <div className="font-black text-slate-900 text-xl">Login</div>
              <div className="text-sm text-slate-800 font-medium">Secure access to diagnostics & patient history</div>
            </div>
          </div>
        </div>

        <div className="p-6">
          <form onSubmit={onSubmit} className="space-y-4">
            <label className="block">
              <div className="text-xs font-bold uppercase text-slate-900 mb-2 flex items-center gap-2">
                <i className="bi bi-envelope-fill" style={{ fontSize: '14px' }} /> Email
              </div>
              <input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                type="email"
                autoComplete="email"
                className="w-full p-3 border border-blue-200 rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/40 text-slate-900"
                placeholder="you@example.com"
              />
            </label>

            <label className="block">
              <div className="text-xs font-bold uppercase text-slate-900 mb-2 flex items-center gap-2">
                <i className="bi bi-lock-fill" style={{ fontSize: '14px' }} /> Password
              </div>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                type="password"
                autoComplete="current-password"
                className="w-full p-3 border border-blue-200 rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/40 text-slate-900"
                placeholder="Enter your password"
              />
            </label>

            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="w-full py-3 rounded-xl font-bold text-white shadow-md transition-all disabled:opacity-50 disabled:cursor-not-allowed bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
            >
              {submitting ? (
                <LoadingSpinner size="sm" text="" />
              ) : (
                <span className="inline-flex items-center gap-2 justify-center">
                  <i className="bi bi-box-arrow-in-right" style={{ fontSize: '18px' }} />
                  Login
                </span>
              )}
            </button>

            <div className="text-sm text-slate-800">
              Don&apos;t have an account?{' '}
              <button
                type="button"
                onClick={onSwitchToSignup}
                className="font-bold text-blue-700 hover:underline"
              >
                Create one
              </button>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
}

