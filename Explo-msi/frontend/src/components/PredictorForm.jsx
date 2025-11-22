import React, { useState } from 'react'

const defaultUCS = {
  density: "2.60",
  porosity: "4.5",
  moisture_content: "0.6",
  p_wave_velocity: "3250",
  schmidt_rebound: "38",
  point_load_index: "1.4",
  depth: "45"
}

const defaultSlope = {
  cohesion: "25",
  friction_angle: "30",
  slope_angle: "40",
  slope_height: "12",
  pore_pressure: "0.1"   // frontend-friendly key -> backend maps to 'pore water pressure ratio'
}

export default function PredictorForm({ setResults }){
  // store as strings to avoid NaN in controlled inputs
  const [ucs, setUcs] = useState(defaultUCS)
  const [slope, setSlope] = useState(defaultSlope)
  const [loading, setLoading] = useState(false)

  // helper: convert string-fields to numbers, validate
  function convertAndValidate(obj, requiredKeys, friendlyName = 'payload') {
    const out = {}
    for (const k of requiredKeys) {
      const raw = obj[k]
      if (raw === undefined || raw === null || String(raw).trim() === "") {
        throw new Error(`${friendlyName} missing value for '${k}'`)
      }
      const num = Number(String(raw).trim())
      if (Number.isNaN(num)) {
        throw new Error(`${friendlyName} field '${k}' is not a valid number: '${raw}'`)
      }
      out[k] = num
    }
    return out
  }

  async function submitBoth(e){
    e.preventDefault()
    setLoading(true)
    try{
      // define required keys (must match Pydantic models / pipeline inputs)
      const ucsKeys = [
        "density","porosity","moisture_content","p_wave_velocity",
        "schmidt_rebound","point_load_index","depth"
      ]
      const slopeKeys = [
        "cohesion","friction_angle","slope_angle","slope_height","pore_pressure"
      ]

      // convert & validate
      const ucsNumeric = convertAndValidate(ucs, ucsKeys, 'UCS')
      const slopeNumeric = convertAndValidate(slope, slopeKeys, 'Slope')

      // Build payload exactly as backend expects (frontend sends pore_pressure; backend maps it)
      const payload = {
        ucs: ucsNumeric,
        slope: slopeNumeric
      }

      const res = await fetch('http://localhost:8000/predict/msi', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })

      if(!res.ok){
        const t = await res.text()
        throw new Error(t || 'Request failed')
      }
      const data = await res.json()
      setResults(data)
    }catch(err){
      // show detailed error to the developer/user
      console.error(err)
      alert(String(err))
    }finally{
      setLoading(false)
    }
  }

  // generic input renderer to keep JSX concise
  function renderInputs(stateObj, setStateFn) {
    return Object.keys(stateObj).map(k => (
      <label key={k} className="block mb-2 text-sm">
        <div className="text-xs text-slate-500">{k}</div>
        <input
          value={stateObj[k]}
          onChange={e => setStateFn(prev => ({...prev, [k]: e.target.value }))}
          className="mt-1 w-full p-2 border rounded"
        />
      </label>
    ))
  }

  return (
    <form onSubmit={submitBoth} className="grid gap-4 grid-cols-1 md:grid-cols-2 mb-6">
      <div className="p-4 bg-white rounded shadow">
        <h2 className="font-semibold mb-2">UCS Features</h2>
        {renderInputs(ucs, setUcs)}
      </div>

      <div className="p-4 bg-white rounded shadow">
        <h2 className="font-semibold mb-2">Slope Features</h2>
        {renderInputs(slope, setSlope)}
      </div>

      <div className="col-span-1 md:col-span-2 flex gap-2">
        <button type="submit" disabled={loading} className="px-4 py-2 bg-blue-600 text-white rounded">
          {loading ? 'Computing...' : 'Compute MSI'}
        </button>

        <button type="button" onClick={async ()=>{ 
          try {
            // quick UCS test (convert strings to numbers first)
            const ucsKeys = ["density","porosity","moisture_content","p_wave_velocity","schmidt_rebound","point_load_index","depth"]
            const ucsNumeric = convertAndValidate(ucs, ucsKeys, "UCS (test)")
            const res = await fetch('http://localhost:8000/predict/ucs', {
              method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify(ucsNumeric)
            })
            if (!res.ok) {
              const t = await res.text()
              throw new Error(t || 'UCS request failed')
            }
            const d = await res.json(); alert('UCS: '+JSON.stringify(d))
          } catch(e) {
            alert(String(e))
          }
        }} className="px-4 py-2 bg-slate-200 rounded">Test UCS</button>
      </div>
    </form>
  )
}
