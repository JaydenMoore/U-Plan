import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

// Custom marker icons based on risk level
const createRiskIcon = (riskScore) => {
  let color = '#10b981' // green for low risk
  if (riskScore > 7) color = '#ef4444' // red for high risk
  else if (riskScore > 4) color = '#f59e0b' // orange for medium risk
  
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })
}

// Component to fit map bounds to markers
const FitBounds = ({ assessments }) => {
  const map = useMap()
  
  useEffect(() => {
    if (assessments && assessments.length > 0) {
      const bounds = assessments.map(a => [a.latitude, a.longitude])
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 10 })
    }
  }, [assessments, map])
  
  return null
}

const DashboardPage = () => {
  const [metrics, setMetrics] = useState(null)
  const [recentAssessments, setRecentAssessments] = useState([])
  const [apiLogs, setApiLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'

  const fetchDashboardData = async () => {
    try {
      // Fetch all dashboard data
      const [metricsRes, assessmentsRes, logsRes] = await Promise.all([
        fetch(`${API_BASE}/dashboard/metrics`),
        fetch(`${API_BASE}/dashboard/recent-assessments?limit=10`),
        fetch(`${API_BASE}/dashboard/api-logs?limit=20`)
      ])

      if (metricsRes.ok) {
        const metricsData = await metricsRes.json()
        setMetrics(metricsData)
      }

      if (assessmentsRes.ok) {
        const assessmentsData = await assessmentsRes.json()
        setRecentAssessments(assessmentsData.assessments || [])
      }

      if (logsRes.ok) {
        const logsData = await logsRes.json()
        setApiLogs(logsData.logs || [])
      }

      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err)
      setError('Failed to load dashboard data. Make sure the backend server is running.')
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDashboardData()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000)
    return () => clearInterval(interval)
  }, [])

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  const getStatusColor = (statusCode) => {
    if (statusCode >= 200 && statusCode < 300) return 'bg-green-100 text-green-800'
    if (statusCode >= 400 && statusCode < 500) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  const getRiskBadge = (score) => {
    if (score > 7) return <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">High Risk</span>
    if (score > 4) return <span className="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800">Medium Risk</span>
    return <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">Low Risk</span>
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading developer dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold mb-4">Dashboard Error</h2>
          <p className="text-gray-400 mb-6">{error}</p>
          <Link to="/" className="text-blue-400 hover:text-blue-300">← Back to Home</Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link to="/" className="text-gray-400 hover:text-white transition">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <div>
                <h1 className="text-2xl font-bold flex items-center">
                  <span className="mr-2">🔧</span>
                  Developer Dashboard
                </h1>
                <p className="text-sm text-gray-400">System Monitoring & Analytics</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button 
                onClick={fetchDashboardData}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition"
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Warning Banner */}
        <div className="bg-yellow-900 border-l-4 border-yellow-500 p-4 mb-6 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-2xl">⚠️</span>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-200">
                <strong>Developer Tool:</strong> This dashboard is for monitoring system performance and debugging. 
                Not intended for end users.
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Cards */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Assessments</p>
                  <p className="text-3xl font-bold mt-1">{metrics.total_assessments?.toLocaleString() || 0}</p>
                </div>
                <div className="text-4xl">📊</div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Requests</p>
                  <p className="text-3xl font-bold mt-1">{metrics.total_requests?.toLocaleString() || 0}</p>
                </div>
                <div className="text-4xl">⚡</div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Success Rate</p>
                  <p className="text-3xl font-bold mt-1">{((metrics.success_rate || 0) * 100).toFixed(2)}%</p>
                </div>
                <div className="text-4xl">
                  {metrics.success_rate >= 0.95 ? '✅' : metrics.success_rate >= 0.80 ? '⚠️' : '❌'}
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">System Health</p>
                  <p className="text-2xl font-bold mt-1">{metrics.system_health || 'Unknown'}</p>
                </div>
                <div className="text-4xl">
                  {metrics.system_health === 'Excellent' ? '💚' : 
                   metrics.system_health === 'Good' ? '💛' : '❤️'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Additional Stats */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <p className="text-gray-400 text-sm mb-2">Avg Risk Score</p>
              <p className="text-2xl font-bold">{metrics.avg_risk_score?.toFixed(2) || '0.00'} / 10</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <p className="text-gray-400 text-sm mb-2">High Risk Locations</p>
              <p className="text-2xl font-bold text-red-400">{metrics.high_risk_locations || 0}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <p className="text-gray-400 text-sm mb-2">Uptime</p>
              <p className="text-2xl font-bold">{metrics.uptime_hours?.toFixed(2) || '0.00'} hours</p>
            </div>
          </div>
        )}

        {/* Assessment Locations Map */}
        <div className="mb-8 bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <div className="p-6 border-b border-gray-700">
            <h2 className="text-xl font-bold flex items-center">
              <span className="mr-2">🗺️</span>
              Assessment Locations
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Showing {recentAssessments.length} recent assessment locations on the map
            </p>
          </div>
          <div className="relative" style={{ height: '500px' }}>
            {recentAssessments.length > 0 ? (
              <MapContainer
                center={[recentAssessments[0]?.latitude || 0, recentAssessments[0]?.longitude || 0]}
                zoom={2}
                style={{ height: '100%', width: '100%' }}
                scrollWheelZoom={true}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <FitBounds assessments={recentAssessments} />
                {recentAssessments.map((assessment, index) => (
                  <Marker
                    key={index}
                    position={[assessment.latitude, assessment.longitude]}
                    icon={createRiskIcon(assessment.overall_risk_score)}
                  >
                    <Popup>
                      <div className="text-gray-900 text-sm">
                        <div className="font-bold mb-2">
                          Risk Score: {assessment.overall_risk_score?.toFixed(2)}
                        </div>
                        <div className="space-y-1 text-xs">
                          <div>
                            <strong>Location:</strong> {assessment.latitude?.toFixed(4)}, {assessment.longitude?.toFixed(4)}
                          </div>
                          <div>
                            <strong>Flood Risk:</strong> {assessment.flood_risk}
                          </div>
                          <div>
                            <strong>Heat Risk:</strong> {assessment.heat_risk}
                          </div>
                          <div>
                            <strong>Air Quality:</strong> {assessment.air_quality_risk}
                          </div>
                          {assessment.timestamp && (
                            <div className="mt-2 pt-2 border-t border-gray-300">
                              <strong>Assessed:</strong> {formatTimestamp(assessment.timestamp)}
                            </div>
                          )}
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </MapContainer>
            ) : (
              <div className="h-full flex items-center justify-center bg-gray-700">
                <div className="text-center">
                  <p className="text-gray-400 mb-2">No assessment locations to display</p>
                  <p className="text-sm text-gray-500">Locations will appear here after assessments are made</p>
                </div>
              </div>
            )}
          </div>
          <div className="p-4 bg-gray-750 border-t border-gray-700">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-green-500 border-2 border-white"></div>
                  <span className="text-gray-400">Low Risk (0-4)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-yellow-500 border-2 border-white"></div>
                  <span className="text-gray-400">Medium Risk (4-7)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 rounded-full bg-red-500 border-2 border-white"></div>
                  <span className="text-gray-400">High Risk (7-10)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Assessments */}
          <div className="bg-gray-800 rounded-lg border border-gray-700">
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-xl font-bold flex items-center">
                <span className="mr-2">🗺️</span>
                Recent Assessments
              </h2>
            </div>
            <div className="p-6">
              {recentAssessments.length > 0 ? (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {recentAssessments.map((assessment, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1">
                          <p className="text-sm text-gray-400">
                            {formatTimestamp(assessment.timestamp)}
                          </p>
                          <p className="text-sm font-mono text-gray-300">
                            {assessment.latitude?.toFixed(4)}, {assessment.longitude?.toFixed(4)}
                          </p>
                        </div>
                        {getRiskBadge(assessment.overall_risk_score)}
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Risk Score:</span>
                        <span className="font-bold">{assessment.overall_risk_score?.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between text-sm mt-1">
                        <span className="text-gray-400">Flood:</span>
                        <span>{assessment.flood_risk}</span>
                      </div>
                      <div className="flex justify-between text-sm mt-1">
                        <span className="text-gray-400">Heat:</span>
                        <span>{assessment.heat_risk}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-center py-8">No recent assessments</p>
              )}
            </div>
          </div>

          {/* API Logs */}
          <div className="bg-gray-800 rounded-lg border border-gray-700">
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-xl font-bold flex items-center">
                <span className="mr-2">📝</span>
                API Call Logs
              </h2>
            </div>
            <div className="p-6">
              {apiLogs.length > 0 ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {apiLogs.map((log, index) => (
                    <div key={index} className="bg-gray-700 rounded p-3 border border-gray-600 text-sm">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-400 text-xs">
                          {formatTimestamp(log.timestamp)}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-bold ${getStatusColor(log.status_code)}`}>
                          {log.status_code}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-mono text-blue-400 font-bold">{log.method}</span>
                        <span className="font-mono text-gray-300">{log.endpoint}</span>
                      </div>
                      <div className="flex justify-between text-xs text-gray-400">
                        <span>Response: {log.response_time_ms?.toFixed(0)}ms</span>
                        {log.client_ip && <span>IP: {log.client_ip}</span>}
                      </div>
                      {log.error_message && (
                        <div className="mt-2 text-xs text-red-400 bg-red-900 bg-opacity-20 p-2 rounded">
                          Error: {log.error_message}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-center py-8">No API logs available</p>
              )}
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold mb-4">📚 Developer Information</h3>
          <div className="grid md:grid-cols-2 gap-6 text-sm">
            <div>
              <h4 className="font-semibold text-blue-400 mb-2">Available API Endpoints:</h4>
              <ul className="space-y-1 text-gray-400 font-mono">
                <li>GET /dashboard/metrics</li>
                <li>GET /dashboard/recent-assessments</li>
                <li>GET /dashboard/system-metrics</li>
                <li>GET /dashboard/api-logs</li>
                <li>GET /dashboard/stats</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-400 mb-2">Dashboard Features:</h4>
              <ul className="space-y-1 text-gray-400">
                <li>✅ Real-time system monitoring</li>
                <li>✅ Performance analytics</li>
                <li>✅ Request/error tracking</li>
                <li>✅ Risk assessment history</li>
                <li>✅ Auto-refresh every 30s</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DashboardPage
