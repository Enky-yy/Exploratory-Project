import React, { useState, useEffect } from 'react'
import PredictorForm from './components/PredictorForm'
import ResultsCard from './components/ResultsCard'
import Charts from './components/Charts'

export default function App(){
  const [results, setResults] = useState(null)

  // demo history (will be appended when you get a real result)
  const [msiHistory, setMsiHistory] = useState([
    { ts: '2025-11-18', msi: 82.1, ucs: 120, slope_prob: 0.12 },
    { ts: '2025-11-19', msi: 78.5, ucs: 110, slope_prob: 0.18 },
    { ts: '2025-11-20', msi: 73.6, ucs: 95, slope_prob: 0.26 }
  ])

  // placeholder feature importances (replace with backend fetch later)
  const [featureImportances, setFeatureImportances] = useState([
    { feature: 'p_wave_velocity', importance: 0.28 },
    { feature: 'point_load_index', importance: 0.22 },
    { feature: 'density', importance: 0.14 },
    { feature: 'schmidt_rebound', importance: 0.10 }
  ])

  // placeholder UCS distribution bins
  const [ucsDistribution, setUcsDistribution] = useState([
    { bin: '0-20', count: 5 }, { bin: '20-40', count: 15 }, { bin: '40-60', count: 45 },
    { bin: '60-80', count: 60 }, { bin: '80-100', count: 30 }, { bin: '100-120', count: 10 }
  ])

  // append new result into history (keeps last 50)
  useEffect(() => {
    if (results) {
      const now = new Date().toISOString().slice(0, 19).replace('T', ' ')
      const newRow = { ts: now, msi: results.msi, ucs: results.ucs, slope_prob: results.slope_failure_prob }
      setMsiHistory(prev => [...prev.slice(-49), newRow])
    }
  }, [results])

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Explo — Mine Safety Index (MSI)</h1>
        <PredictorForm setResults={setResults} />
        {results && <ResultsCard results={results} />}
        <Charts msiHistory={msiHistory} featureImportances={featureImportances} ucsDistribution={ucsDistribution} />
      </div>
    </div>
  )
}
