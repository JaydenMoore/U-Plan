import React from 'react'

const ResultsCard = ({ assessment }) => {
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'Medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'Low':
        return 'text-green-600 bg-green-50 border-green-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskIcon = (risk) => {
    switch (risk) {
      case 'High':
        return '⚠️'
      case 'Medium':
        return '⚡'
      case 'Low':
        return '✅'
      default:
        return '❓'
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
              <span className="font-bold">{assessment.overall_risk_score}/10</span>
            </div>
          </div>
        )}
      </div>

      {/* Air Quality Details */}
      {assessment.air_quality_index && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <h3 className="font-semibold text-slate-800 mb-3 flex items-center">
            🏭 Air Quality Details
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white rounded-lg p-2 text-center shadow-sm">
              <div className="text-2xl font-bold text-purple-600">
                {assessment.air_quality_index}
              </div>
              <div className="text-xs text-gray-500 mt-1">scale 0 - 500</div>
              <div className="text-sm font-medium text-gray-700">📊 AQI</div>
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
        </div>
      )}

      {/* AI Summary */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="font-semibold text-purple-800 mb-2">🧠 Planning Summary</h3>
        <p className="text-sm text-purple-700 leading-relaxed">
          {assessment.summary}
        </p>
      </div>

      {/* Data Source */}
      <div className="text-xs text-gray-500 text-center pt-4 border-t">
        <p>Data source: NASA POWER API</p>
        <p>5-year climate average (2019-2023)</p>
      </div>
    </div>
  )
}

export default ResultsCard