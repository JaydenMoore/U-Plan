import React, { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import Map from './Map'
import ResultsCard from './ResultsCard'

const MapPage = () => {
  const [assessment, setAssessment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showResults, setShowResults] = useState(false)
  const abortControllerRef = useRef(null)

  const handleLocationClick = async (lat, lng) => {
    console.log('MapPage: handleLocationClick called with:', lat, lng)
    
    // Cancel any previous request
    if (abortControllerRef.current) {
      console.log('MapPage: Cancelling previous request')
      abortControllerRef.current.abort()
    }
    
    // Create new abort controller for this request
    abortControllerRef.current = new AbortController()
    
    // Clear previous assessment data immediately to prevent stale data
    setAssessment(null)
    setLoading(true)
    setError('')
    setShowResults(true)
    
    try {
      console.log('MapPage: Making API request to assess-risk endpoint')
      const response = await fetch('http://localhost:8001/assess-risk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ latitude: lat, longitude: lng }),
        signal: abortControllerRef.current.signal
      })
      
      console.log('MapPage: API response status:', response.status)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      console.log('MapPage: NASA data received:', data)
      
      // Only update if this request wasn't cancelled
      if (!abortControllerRef.current.signal.aborted) {
        setAssessment(data)
      }
    } catch (err) {
      // Ignore abort errors
      if (err.name === 'AbortError') {
        console.log('MapPage: Request was cancelled')
        return
      }
      
      console.error('MapPage: Error fetching assessment:', err)
      let errorMessage = 'Failed to fetch environmental data. '
      
      if (err.message.includes('Failed to fetch')) {
        errorMessage += 'Please make sure the backend server is running on port 8001.'
      } else {
        errorMessage += err.message
      }
      
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-900 relative overflow-hidden">
      {/* Top Navigation Bar - Centered Chunk */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-[2000] bg-black bg-opacity-70 backdrop-blur-sm rounded-lg px-6 py-3">
        <div className="flex items-center space-x-4">
          <Link 
            to="/" 
            className="flex items-center text-white hover:text-blue-300 transition-colors duration-200"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <span className="text-sm font-medium">Home</span>
          </Link>
          
          <div className="flex items-center text-white">
            <span className="text-lg mr-2">🌍</span>
            <h1 className="text-lg font-semibold">Urban Planner AI</h1>
          </div>
        </div>
      </div>

      {/* Full-Screen Map */}
      <div className="flex-1 relative">
        <Map onLocationClick={handleLocationClick} assessment={assessment} />
      </div>

      {/* Sliding Results Panel */}
      <div 
        className={`absolute top-4 right-0 h-[calc(100vh-2rem)] w-96 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out z-[1600] rounded-l-lg ${
          showResults ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          {/* Results Header */}
          <div className="p-4 border-b bg-gray-50">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-800">Risk Assessment</h2>
              <button
                onClick={() => setShowResults(false)}
                className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <p className="text-gray-600 text-sm mt-1">
              Environmental data and recommendations
            </p>
          </div>
          
          {/* Results Content */}
          <div className="flex-1 p-4 overflow-y-auto">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Fetching NASA data...</span>
              </div>
            )}
            
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                  <div className="text-red-400">⚠️</div>
                  <div className="ml-3">
                    <p className="text-red-800 font-medium">Error</p>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                </div>
              </div>
            )}
            
            {assessment && !loading && (
              <ResultsCard assessment={assessment} />
            )}
            
            {!assessment && !loading && !error && (
              <div className="text-center py-12 text-gray-500">
                <div className="text-4xl mb-4">🗺️</div>
                <p className="text-lg font-medium">Urban Planner AI</p>
                <p className="text-sm mt-2">
                  Click on the map to start analyzing environmental risks for any location
                </p>
                <div className="mt-6 text-xs text-gray-400">
                  <p>Powered by NASA POWER API</p>
                  <p>Real-time climate data analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Toggle Button (for mobile/when panel is hidden) */}
      {!showResults && (assessment || loading || error) && (
        <button
          onClick={() => setShowResults(true)}
          className="absolute top-4 right-4 z-[1700] bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg shadow-lg transition-all duration-200 flex items-center space-x-2"
        >
          <span className="text-sm font-medium">📊 View Results</span>
        </button>
      )}
    </div>
  )
}

export default MapPage