import axios from 'axios';

// Connect to your Node.js Backend (Port 5000)
const API = axios.create({ baseURL: 'http://localhost:5000/api' });

export const analyzeImage = async (moduleName, file, metadata = {}) => {
  const formData = new FormData();
  formData.append('module', moduleName);
  formData.append('image', file);
  
  // Add metadata (Age, Sex, etc.) if it exists
  Object.keys(metadata).forEach(key => {
    formData.append(key, metadata[key]);
  });

  try {
    const response = await API.post('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};

export const getHistory = async () => {
  try {
    const response = await API.get('/history');
    return response.data;
  } catch (error) {
    console.error("History Error:", error);
    return [];
  }
};