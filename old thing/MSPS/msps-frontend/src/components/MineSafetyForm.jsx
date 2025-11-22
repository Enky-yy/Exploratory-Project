import React, { useState } from "react";

export default function MineSafetyForm() {
  const [formData, setFormData] = useState({
    cohesion: "",
    friction_angle: "",
    slope_angle: "",
    slope_height: "",
    PoreWaterPressureRatio: "",
    density: "",
    porosity: "",
    moisture_content: "",
    p_wave_velocity: "",
    schmidt_rebound_value: "",
    lithology: "",
    point_load_index: "",
    depth: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(formData).map(([k, v]) => [k, parseFloat(v)])
          )
        ),
      });

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Error during prediction. Check console for details.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 shadow-lg rounded-xl bg-white mt-10">
      <h2 className="text-2xl font-bold text-center mb-6">Mine Safety Prediction</h2>

      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label className="capitalize text-sm">{key.replace(/_/g, " ")}</label>
            <input
              type="number"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="border rounded-md w-full p-2"
              step="any"
              required
            />
          </div>
        ))}
        <div className="col-span-2 flex justify-center mt-4">
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg"
          >
            {loading ? "Predicting..." : "Predict Safety"}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-6 bg-gray-100 p-4 rounded-lg text-center">
          <p><strong>Predicted UCS:</strong> {result.Predicted_UCS}</p>
          <p><strong>Factor of Safety (FOS):</strong> {result.Predicted_FOS}</p>
          <p className="text-lg mt-2">
            <strong>Status:</strong> {result.Failure_Status}
          </p>
        </div>
      )}
    </div>
  );
}
