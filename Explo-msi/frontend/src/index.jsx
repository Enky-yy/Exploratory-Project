// frontend/src/index.jsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css';
import './index.css'      // <--- Tailwind directives processed here

createRoot(document.getElementById('root')).render(<App />)
