// frontend/src/components/Charts.jsx
import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area
} from "recharts";

/**
 * Props:
 *  - msiHistory: [{ts: '2025-11-21 10:00', msi: 72.3, ucs: 90.2, slope_prob: 0.24}, ...]
 *  - featureImportances: [{feature: 'p_wave_velocity', importance: 0.32}, ...]
 *  - ucsDistribution: [{bin: '0-20', count: 5}, ...]
 */
export default function Charts({ msiHistory = [], featureImportances = [], ucsDistribution = [] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
      {/* MSI Trend */}
      <div className="p-4 bg-white rounded shadow">
        <h4 className="font-semibold mb-2">MSI Trend</h4>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <LineChart data={msiHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ts" tick={{fontSize:11}} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="msi" name="MSI" stroke="#2563EB" strokeWidth={2} dot={{r:2}} />
              <Line type="monotone" dataKey="ucs" name="UCS (MPa)" stroke="#10B981" strokeWidth={2} />
              <Line type="monotone" dataKey="slope_prob" name="Slope Prob" stroke="#EF4444" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="p-4 bg-white rounded shadow">
        <h4 className="font-semibold mb-2">Feature Importance</h4>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={featureImportances} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" width={150} />
              <Tooltip />
              <Bar dataKey="importance" name="Importance" fill="#6366F1" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* UCS Distribution (Area) */}
      <div className="p-4 bg-white rounded shadow md:col-span-2">
        <h4 className="font-semibold mb-2">UCS Distribution</h4>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <AreaChart data={ucsDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="count" stroke="#06B6D4" fill="#06B6D4" fillOpacity={0.15}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
