import React, { useState } from 'react'

const ResultsCard = ({ assessment }) => {
  const [feedback, setFeedback] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [submitSuccess, setSubmitSuccess] = useState(false)
  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

  const getRiskColor = (risk) => {
    switch (risk) {
      // Standard risk levels (for flood and heat)
      case 'High':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'Medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'Low':
        return 'text-green-600 bg-green-50 border-green-200'
      
      // EPA AQI categories (for air quality)
      case 'Good':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'Moderate':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'Unhealthy for Sensitive Groups':
        return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'Unhealthy':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'Very Unhealthy':
        return 'text-purple-600 bg-purple-50 border-purple-200'
      case 'Hazardous':
        return 'text-red-800 bg-red-100 border-red-300'
      
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskIcon = (risk) => {
    switch (risk) {
      // Standard risk levels (for flood and heat)
      case 'High':
        return '⚠️'
      case 'Medium':
        return '⚡'
      case 'Low':
        return '✅'
      
      // EPA AQI categories (for air quality)
      case 'Good':
        return '✅'
      case 'Moderate':
        return '⚡'
      case 'Unhealthy for Sensitive Groups':
        return '⚠️'
      case 'Unhealthy':
        return '🔴'
      case 'Very Unhealthy':
        return '🚨'
      case 'Hazardous':
        return '☢️'
      
      default:
        return '❓'
    }
  }

  const formatSummary = (summaryText) => {
    if (!summaryText) return null;

    // Split by newline, then process each line
    return summaryText.split('\n').map((line, index) => {
      // Trim the line to handle spaces
      const trimmedLine = line.trim();

      // Handle list items starting with '*'
      if (trimmedLine.startsWith('* ')) {
        const content = trimmedLine.substring(2);
        // Process for bold text
        const parts = content.split(/(\*\*.*?\*\*)/g);
        return (
          <li key={index} className="ml-4 list-disc">
            {parts.map((part, i) => 
              part.startsWith('**') && part.endsWith('**') ? 
              <strong key={i}>{part.slice(2, -2)}</strong> : 
              part
            )}
          </li>
        );
      }

      // Handle bold text in regular lines
      const parts = trimmedLine.split(/(\*\*.*?\*\*)/g);
      return (
        <p key={index}>
          {parts.map((part, i) => 
            part.startsWith('**') && part.endsWith('**') ? 
            <strong key={i}>{part.slice(2, -2)}</strong> : 
            part
          )}
        </p>
      );
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!feedback.trim()) return
    setSubmitting(true)
    setSubmitError('')
    setSubmitSuccess(false)
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: assessment.latitude,
          longitude: assessment.longitude,
          summary: assessment.summary,
          overall_risk_score: assessment.overall_risk_score,
          feedback,
          source: 'frontend'
        })
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || 'Failed to submit feedback')
      }
      setSubmitSuccess(true)
      setFeedback('')
    } catch (err) {
      setSubmitError(err.message || 'Failed to submit feedback')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Location Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-semibold text-blue-800 mb-2">📍 Location</h3>
        <div className="text-sm text-blue-700">
          <p>Latitude: {assessment.latitude.toFixed(4)}°</p>
          <p>Longitude: {assessment.longitude.toFixed(4)}°</p>
          <p className="text-xs text-blue-600 mt-1">
            Last updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="space-y-3">
        <h3 className="font-semibold text-gray-800">⚡ Risk Assessment</h3>
        
        {/* Flood Risk */}
        <div className={`border rounded-lg p-3 ${getRiskColor(assessment.flood_risk)}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-lg">{getRiskIcon(assessment.flood_risk)}</span>
              <span className="font-medium">Flood Risk</span>
            </div>
            <span className="font-bold">{assessment.flood_risk}</span>
          </div>
        </div>

        {/* Heat Risk */}
        <div className={`border rounded-lg p-3 ${getRiskColor(assessment.heat_risk)}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-lg">{getRiskIcon(assessment.heat_risk)}</span>
              <span className="font-medium">Heat Risk</span>
            </div>
            <span className="font-bold">{assessment.heat_risk}</span>
          </div>
        </div>

        {/* Air Quality Risk */}
        {assessment.air_quality_risk && (
          <div className={`border rounded-lg p-3 ${getRiskColor(assessment.air_quality_risk)}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-lg">{getRiskIcon(assessment.air_quality_risk)}</span>
                <span className="font-medium">Air Quality Risk</span>
              </div>
              <span className="font-bold">{assessment.air_quality_risk}</span>
            </div>
          </div>
        )}

        {/* Overall Risk Score */}
        {assessment.overall_risk_score && (
          <div className={`border rounded-lg p-3 ${
            assessment.overall_risk_score >= 7 ? 'text-red-600 bg-red-50 border-red-200' :
            assessment.overall_risk_score >= 5 ? 'text-orange-600 bg-orange-50 border-orange-200' :
            assessment.overall_risk_score >= 3 ? 'text-yellow-600 bg-yellow-50 border-yellow-200' :
            'text-green-600 bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-lg">🏆</span>
                <span className="font-medium">Overall Risk Score</span>
              </div>
              <span className="font-bold">{assessment.overall_risk_score.toFixed(2)}/10</span>
            </div>
          </div>
        )}

        {/* Probabilistic Risk Assessment */}
        {assessment.probabilistic_risk && (
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-300 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 mb-3 flex items-center">
              🎯 Advanced Risk Analysis
            </h4>
            
            {assessment.probabilistic_risk.model_fitted ? (
              <div className="space-y-3">
                {/* Expected Risk */}
                <div className="bg-white rounded-lg p-3 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">📊 Expected Risk Level</span>
                    <span className="text-lg font-bold text-purple-600">
                      {assessment.probabilistic_risk.expected_risk.toFixed(1)}
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">
                    Probabilistic assessment incorporating hazard dependencies
                  </div>
                </div>

                {/* Confidence Interval */}
                {assessment.risk_confidence_interval && (
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">📈 Confidence Range</span>
                      <span className="text-sm font-bold text-indigo-600">
                        {assessment.risk_confidence_interval.lower_bound.toFixed(1)} - {assessment.risk_confidence_interval.upper_bound.toFixed(1)}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 mb-1">
                      90% confidence interval • Median: {assessment.risk_confidence_interval.median.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-500">
                      Range shows uncertainty in risk estimates
                    </div>
                  </div>
                )}

                {/* Water Risk Assessment */}
                {assessment.water_risk && (
                  <div className="bg-white rounded-lg p-3 shadow-sm border border-blue-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">💧 Water Risk Assessment</span>
                      <span className={`text-sm font-bold px-2 py-1 rounded ${
                        assessment.water_risk.overall_category === 'Extremely High' ? 'text-red-700 bg-red-100' :
                        assessment.water_risk.overall_category === 'High' ? 'text-orange-700 bg-orange-100' :
                        assessment.water_risk.overall_category === 'Medium-High' ? 'text-yellow-700 bg-yellow-100' :
                        assessment.water_risk.overall_category === 'Low-Medium' ? 'text-blue-700 bg-blue-100' :
                        'text-green-700 bg-green-100'
                      }`}>
                        {assessment.water_risk.overall_category}
                      </span>
                    </div>
                    
                    {/* Water Risk Details */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Water Stress:</span>
                        <span className="font-medium">{assessment.water_risk.baseline_water_stress}/5</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Drought Risk:</span>
                        <span className="font-medium">{assessment.water_risk.drought_risk}/5</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Flood Risk:</span>
                        <span className="font-medium">{assessment.water_risk.flood_risk_score}/5</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Overall:</span>
                        <span className="font-medium">{assessment.water_risk.overall_water_risk}/5</span>
                      </div>
                    </div>
                    
                    <div className="text-xs text-gray-600 mt-2 pt-2 border-t border-gray-100">
                      📍 {assessment.water_risk.country} • Source: {assessment.water_risk.source}
                    </div>
                  </div>
                )}

                {/* Model Uncertainty */}
                <div className="bg-gray-50 rounded-lg p-2">
                  <div className="text-xs font-medium text-gray-700 mb-1">
                    Model Confidence: <span className={`font-bold ${
                      assessment.model_uncertainty === 'Low' ? 'text-green-600' :
                      assessment.model_uncertainty === 'High' ? 'text-orange-600' :
                      'text-gray-600'
                    }`}>{assessment.model_uncertainty}</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    Based on {assessment.probabilistic_risk.n_samples || 500} Monte Carlo samples
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-sm text-gray-600">
                  ℹ️ Enhanced probabilistic analysis not available due to insufficient historical data.
                  Using simplified risk assessment.
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* AI Summary */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="font-semibold text-purple-800 mb-2">🧠 AI Planning Summary</h3>
        <p className="text-sm text-purple-700 leading-relaxed">
          {formatSummary(assessment.summary)}
        </p>
      </div>

      {/* AI Model Explanation */}
      {assessment.model_explanation && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-3">🤖 AI Model Explanation</h3>
          
          {/* Prediction Details */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="text-sm font-medium text-gray-700 mb-1">Prediction Value</div>
              <div className="text-lg font-bold text-blue-600">
                {assessment.model_explanation.prediction?.toFixed(2)}
              </div>
            </div>
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="text-sm font-medium text-gray-700 mb-1">Confidence Level</div>
              <div className={`text-lg font-bold ${
                assessment.prediction_confidence === 'High' ? 'text-green-600' :
                assessment.prediction_confidence === 'Medium' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {assessment.prediction_confidence || 'Unknown'}
              </div>
            </div>
          </div>

          {/* Feature Importance */}
          {assessment.feature_importance && assessment.feature_importance.length > 0 && (
            <div className="bg-white rounded-lg p-3 shadow-sm mb-3">
              <div className="text-sm font-medium text-gray-700 mb-3">🎯 Feature Importance</div>
              <div className="space-y-2">
                {assessment.feature_importance.map((feature, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className={`w-2 h-2 rounded-full ${
                        feature.direction === 'increases' ? 'bg-red-400' : 'bg-green-400'
                      }`}></span>
                      <span className="text-sm text-gray-700 capitalize">
                        {feature.feature.replace('_', ' ')}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        feature.direction === 'increases' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                      }`}>
                        {feature.direction} risk
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            feature.direction === 'increases' ? 'bg-red-400' : 'bg-green-400'
                          }`}
                          style={{ width: `${Math.min(feature.importance * 50, 100)}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 w-12 text-right">
                        {(feature.importance * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Natural Language Explanation */}
          {assessment.model_explanation.explanation_text && (
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="text-sm font-medium text-gray-700 mb-2">📝 Detailed Explanation</div>
              <div className="text-sm text-gray-600 leading-relaxed whitespace-pre-line">
                {assessment.model_explanation.explanation_text}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===== DATA SECTIONS (Organized by Category) ===== */}

      {/* Climate Data */}
      <div className="bg-gradient-to-r from-blue-50 to-green-50 border border-gray-200 rounded-lg p-4">
        <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
          🌡️ NASA Climate Data (5-year average)
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-3 text-center shadow-sm">
            <div className="text-3xl font-bold text-blue-600">
              {assessment.rainfall_mm}
            </div>
            <div className="text-xs text-gray-500 mt-1">mm/month</div>
            <div className="text-sm font-medium text-gray-700">💧 Rainfall</div>
          </div>
          <div className="bg-white rounded-lg p-3 text-center shadow-sm">
            <div className="text-3xl font-bold text-orange-600">
              {assessment.temperature_c}°C
            </div>
            <div className="text-xs text-gray-500 mt-1">average</div>
            <div className="text-sm font-medium text-gray-700">🌡️ Temperature</div>
          </div>
        </div>
      </div>

      {/* ===== FLOOD DATA SECTION ===== */}
      
      {/* NASA MODIS Flood Data */}
      {(assessment.flood_risk_score !== null && assessment.flood_risk_score !== undefined) && (
        <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4">
          <h3 className="font-semibold text-cyan-800 mb-3 flex items-center">
            🌊 NASA MODIS Flood Risk (Real-time)
          </h3>
          <div className="grid grid-cols-1 gap-3">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">📡 Real-time Risk Score</span>
                <span className={`text-lg font-bold px-2 py-1 rounded ${
                  assessment.flood_risk_score >= 0.5 ? 'text-red-700 bg-red-100' :
                  assessment.flood_risk_score >= 0.2 ? 'text-orange-700 bg-orange-100' :
                  assessment.flood_risk_score > 0 ? 'text-yellow-700 bg-yellow-100' :
                  'text-green-700 bg-green-100'
                }`}>
                  {(assessment.flood_risk_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-xs text-gray-600 mb-2">
                {assessment.flood_message || 'No flood data available'}
              </div>
              <div className="text-xs text-gray-500">
                Percentage of satellite pixels showing active flooding right now
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Historical Flooding (Global Flood Database) */}
      {(assessment.historic_flood_frequency !== null && assessment.historic_flood_frequency !== undefined) && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-3 flex items-center">
            📊 Historical Flooding (2000–2018)
          </h3>
          <div className="grid grid-cols-1 gap-3">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">🕒 Annual Flood Probability</span>
                <span className={`text-lg font-bold px-2 py-1 rounded ${
                  assessment.historic_flood_frequency >= 0.2 ? 'text-red-700 bg-red-100' :
                  assessment.historic_flood_frequency >= 0.05 ? 'text-orange-700 bg-orange-100' :
                  assessment.historic_flood_frequency > 0 ? 'text-yellow-700 bg-yellow-100' :
                  'text-green-700 bg-green-100'
                }`}>
                  {(assessment.historic_flood_frequency * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-xs text-gray-600 mb-2">
                {assessment.historic_flood_category || 'Historical flood pattern analysis'}
              </div>
              <div className="text-xs text-gray-500">
                Likelihood of flooding based on climate data and geographic factors
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comprehensive Flood Risk Assessment */}
      {assessment.comprehensive_flood_risk && (
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-300 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
            🌊📊 Combined Flood Risk Assessment
          </h3>
          <div className="grid grid-cols-1 gap-3">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">⚡ Overall Flood Risk</span>
                <span className={`text-xl font-bold px-3 py-1 rounded ${
                  assessment.comprehensive_flood_risk.risk_level === 'Very High' ? 'text-red-700 bg-red-100' :
                  assessment.comprehensive_flood_risk.risk_level === 'High' ? 'text-orange-700 bg-orange-100' :
                  assessment.comprehensive_flood_risk.risk_level === 'Moderate' ? 'text-yellow-700 bg-yellow-100' :
                  assessment.comprehensive_flood_risk.risk_level === 'Low' ? 'text-blue-700 bg-blue-100' :
                  'text-green-700 bg-green-100'
                }`}>
                  {assessment.comprehensive_flood_risk.combined_score ? 
                    `${(assessment.comprehensive_flood_risk.combined_score * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="text-sm font-medium text-gray-800 mb-2">
                Risk Level: {assessment.comprehensive_flood_risk.risk_level}
              </div>
              <div className="text-xs text-gray-600 mb-3">
                {assessment.comprehensive_flood_risk.explanation}
              </div>
              
              {/* Breakdown of risk components */}
              <div className="bg-gray-50 rounded p-2 mb-2">
                <div className="text-xs font-medium text-gray-700 mb-1">Risk Composition:</div>
                <div className="flex justify-between text-xs text-gray-600">
                  <span>Real-time (60%): {assessment.flood_percentage_explanation?.realtime_pct}</span>
                  <span>Historical (40%): {assessment.flood_percentage_explanation?.historical_pct}</span>
                </div>
              </div>
              
              <div className="text-xs text-gray-500">
                {assessment.comprehensive_flood_risk.geographic_context}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== AIR QUALITY DATA SECTION ===== */}

            {/* Air Quality Details */}
      {assessment.air_quality_index && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <h3 className="font-semibold text-slate-800 mb-3 flex items-center">
            🏭 Air Quality Details
          </h3>
          <div className="grid grid-cols-1 gap-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                <div className={`text-2xl font-bold ${
                  assessment.air_quality_index >= 301 ? 'text-rose-800' :
                  assessment.air_quality_index >= 201 ? 'text-purple-600' :
                  assessment.air_quality_index >= 151 ? 'text-red-600' :
                  assessment.air_quality_index >= 101 ? 'text-orange-500' :
                  assessment.air_quality_index >= 51 ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {assessment.air_quality_index}
                </div>
                <div className={`text-xs mt-1 font-medium ${
                  assessment.air_quality_index >= 301 ? 'text-rose-800' :
                  assessment.air_quality_index >= 201 ? 'text-purple-600' :
                  assessment.air_quality_index >= 151 ? 'text-red-600' :
                  assessment.air_quality_index >= 101 ? 'text-orange-500' :
                  assessment.air_quality_index >= 51 ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {assessment.air_quality_index >= 301 ? 'Hazardous' :
                   assessment.air_quality_index >= 201 ? 'Very Unhealthy' :
                   assessment.air_quality_index >= 151 ? 'Unhealthy' :
                   assessment.air_quality_index >= 101 ? 'Unhealthy for Sensitive Groups' :
                   assessment.air_quality_index >= 51 ? 'Moderate' :
                   'Good'}
                </div>
                <div className="text-sm font-medium text-gray-700 mt-1">📊 US EPA AQI</div>
              </div>
              <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                <div className="text-2xl font-bold text-indigo-600">
                  {assessment.pm2_5?.toFixed(1)}
                </div>
                <div className="text-xs text-gray-500 mt-1">μg/m³</div>
                <div className="text-sm font-medium text-gray-700">💨 PM2.5</div>
              </div>
              {assessment.pm10 && (
                <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                  <div className="text-xl font-bold text-blue-600">
                    {assessment.pm10?.toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">μg/m³</div>
                  <div className="text-sm font-medium text-gray-700">🌫️ PM10</div>
                </div>
              )}
              {assessment.no2 && (
                <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                  <div className="text-xl font-bold text-green-600">
                    {assessment.no2?.toFixed(0)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">μg/m³</div>
                  <div className="text-sm font-medium text-gray-700">🚗 NO₂</div>
                </div>
              )}
            </div>
            
            {assessment.pm2_5_raw && (
              <div className="text-xs text-gray-500 mt-1 text-center">
                Raw PM2.5: {assessment.pm2_5_raw?.toFixed(1)} μg/m³ | Calibrated: {assessment.pm2_5?.toFixed(1)} μg/m³
              </div>
            )}
          </div>
        </div>
      )}

      {/* ===== WATER RISK DATA SECTION ===== */}

      {/* Water Risk Assessment */}
      {assessment.water_risk && (
        <div className="bg-teal-50 border border-teal-200 rounded-lg p-4">
          <h3 className="font-semibold text-teal-800 mb-3 flex items-center">
            💧 Water Risk Assessment
          </h3>
          <div className="bg-white rounded-lg p-3 shadow-sm border border-blue-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">� Overall Risk Level</span>
              <span className={`text-sm font-bold px-2 py-1 rounded ${
                assessment.water_risk.overall_category === 'Extremely High' ? 'text-red-700 bg-red-100' :
                assessment.water_risk.overall_category === 'High' ? 'text-orange-700 bg-orange-100' :
                assessment.water_risk.overall_category === 'Medium-High' ? 'text-yellow-700 bg-yellow-100' :
                assessment.water_risk.overall_category === 'Low-Medium' ? 'text-blue-700 bg-blue-100' :
                'text-green-700 bg-green-100'
              }`}>
                {assessment.water_risk.overall_category}
              </span>
            </div>
            
            {/* Water Risk Details */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Water Stress:</span>
                <span className="font-medium">{assessment.water_risk.baseline_water_stress}/5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Drought Risk:</span>
                <span className="font-medium">{assessment.water_risk.drought_risk}/5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Flood Risk:</span>
                <span className="font-medium">{assessment.water_risk.flood_risk_score}/5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Overall:</span>
                <span className="font-medium">{assessment.water_risk.overall_water_risk}/5</span>
              </div>
            </div>
            
            <div className="text-xs text-gray-600 mt-2 pt-2 border-t border-gray-100">
              📍 {assessment.water_risk.country} • Source: {assessment.water_risk.source}
            </div>
          </div>
        </div>
      )}

      {/* ===== POPULATION DATA SECTION ===== */}

      {/* Population Data */}
      {assessment.population_density !== null && assessment.population_density !== undefined && (
        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
          <h3 className="font-semibold text-orange-800 mb-3 flex items-center">
            👥 Population Data (GPW-v4)
          </h3>
          <div className="grid grid-cols-1 gap-3">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">🎯 Point Density</span>
                <span className="text-lg font-bold text-orange-600">
                  {assessment.population_density?.toFixed(1)} people/km²
                </span>
              </div>
              <div className="text-xs text-gray-500">
                {assessment.population_density > 1000 ? '🏙️ Very High Density' :
                 assessment.population_density > 500 ? '🏘️ High Density' :
                 assessment.population_density > 150 ? '🏡 Medium Density' :
                 assessment.population_density > 50 ? '🌲 Low Density' :
                 assessment.population_density > 1 ? '🌿 Very Low Density' :
                 '🌍 Uninhabited'}
              </div>
            </div>
            {assessment.population_stats && !assessment.population_stats.error && (
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-sm font-medium text-gray-700 mb-2">📊 Area Statistics (5km radius)</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Average:</span>
                    <span className="font-semibold text-orange-600 ml-1">
                      {assessment.population_stats.mean_density?.toFixed(1)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Peak:</span>
                    <span className="font-semibold text-orange-600 ml-1">
                      {assessment.population_stats.max_density?.toFixed(1)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Populated pixels:</span>
                    <span className="font-semibold text-orange-600 ml-1">
                      {assessment.population_stats.populated_pixels}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Category:</span>
                    <span className="font-semibold text-orange-600 ml-1">
                      {assessment.population_stats.population_category}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Feedback Form */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h3 className="font-semibold text-gray-800 mb-2">💬 Spot a Change? Report It.</h3>
        <form onSubmit={handleSubmit} className="space-y-2">
          <textarea
            className="w-full border border-gray-300 rounded p-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-300"
            rows={3}
            placeholder="Tell us if this summary was helpful or what to improve..."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <button
              type="submit"
              disabled={submitting || !feedback.trim()}
              className={`px-3 py-2 rounded text-white text-sm ${submitting || !feedback.trim() ? 'bg-gray-400' : 'bg-purple-600 hover:bg-purple-700'}`}
            >
              {submitting ? 'Submitting…' : 'Submit'}
            </button>
            {submitSuccess && (
              <span className="text-green-600 text-sm">Thanks! Feedback saved.</span>
            )}
            {submitError && (
              <span className="text-red-600 text-sm">{submitError}</span>
            )}
          </div>
        </form>
      </div>

      {/* Data Source */}
      <div className="text-xs text-gray-500 text-center pt-4 border-t">
        <p>Climate: NASA POWER API (5-year average 2019-2023)</p>
        <p>Air Quality: OpenWeatherMap API</p>
        <p>Population: GPW-v4 (NASA/CIESIN 2020)</p>
        <p>Flood Risk: NASA MODIS (Near real-time)</p>
        <p>Historic Floods: Global Flood Database (GFD, 2000-2018)</p>
        <p>Water Risk: WRI Aqueduct 4.0 Global Maps</p>
        <p>Risk Modeling: Gaussian Copula with Monte Carlo simulation</p>
        {assessment.model_explanation && (
          <p>AI Interpretability: SHAP (SHapley Additive exPlanations)</p>
        )}
      </div>
    </div>
  )
}

export default ResultsCard