import React from 'react'

function riskBand(msi){
  if(msi >= 75) return {label:'Low Risk', color:'bg-green-200'}
  if(msi >= 45) return {label:'Moderate Risk', color:'bg-yellow-200'}
  return {label:'High Risk', color:'bg-red-200'}
}

export default function ResultsCard({ results }){
  const band = riskBand(results.msi)
  return (
    <div className="p-4 bg-white rounded shadow mt-4">
      <h3 className="text-xl font-semibold mb-2">MSI Result</h3>
      <div className={`inline-block px-3 py-1 rounded ${band.color} mr-2`}>{band.label}</div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <div><strong>UCS (MPa)</strong><div>{results.ucs}</div></div>
        <div><strong>Slope Failure Prob</strong><div>{(results.slope_failure_prob*100).toFixed(2)}%</div></div>
        <div><strong>MSI</strong><div className="text-2xl font-bold">{results.msi}</div></div>
        <div><strong>Weights</strong><div>{JSON.stringify(results.weights)}</div></div>
      </div>
    </div>
  )
}
