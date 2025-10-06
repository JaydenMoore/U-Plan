import React, { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Map from './Map'
import ResultsCard from './ResultsCard'

const MapPage = () => {
  const [assessment, setAssessment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showResults, setShowResults] = useState(false)
  const abortControllerRef = useRef(null)
  const mapRef = useRef(null)

  const handleLocationClick = async (lat, lng) => {
    console.log('MapPage: handleLocationClick called with:', lat, lng)
    
    // Normalize longitude to be within -180 to 180 degrees
    // Handle very large positive or negative values properly
    let normalizedLng = lng
    while (normalizedLng > 180) {
      normalizedLng -= 360
    }
    while (normalizedLng < -180) {
      normalizedLng += 360
    }
    console.log('MapPage: Normalized longitude:', normalizedLng)
    
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
      console.log('MapPage: Making API request to assess-location-enhanced endpoint')
      const response = await fetch('https://uplan-backend-156683368413.asia-southeast1.run.app/assess-location-enhanced', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ latitude: lat, longitude: normalizedLng }),
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
        // Notify the map component that assessment is complete
        if (mapRef.current && mapRef.current.completeAssessment) {
          mapRef.current.completeAssessment()
        }
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
        errorMessage += 'Please make sure the backend server is running on port uplan-backend GCloud.'
      } else {
        errorMessage += err.message
      }
      
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-800 relative overflow-hidden">
      {/* Dynamic Island Navigation */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-[2000]">
        <div className="bg-white bg-opacity-80 backdrop-blur-lg rounded-full px-6 py-3 shadow-2xl border border-white border-opacity-20">
          <div className="flex items-center justify-between space-x-8">
            <Link 
              to="/" 
              className="flex items-center text-white hover:text-blue-300 transition-all duration-300 hover:scale-105"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="black" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              <span className="text-sm font-medium text-black">Home</span>
            </Link>
            
            <div className="flex items-center text-white">
              <span className="text-lg mr-2">🌍</span>
              <h1 className="text-lg font-semibold text-black bg-clip-text">
                U-Plan
              </h1>
            </div>

            <div className="w-16"></div> {/* Spacer for centering */}
          </div>
        </div>
      </div>

      {/* Full-Screen Map */}
      <div className="flex-1 relative">
        <Map 
          ref={mapRef}
          onLocationClick={handleLocationClick} 
          assessment={assessment}
          onToggleResults={() => setShowResults(!showResults)}
        />
      </div>

      {/* Sliding Results Panel */}
      <div
        className={`absolute top-0 right-0 h-full bg-white border-l shadow-lg transition-transform duration-300 ease-in-out z-[1800] ${
          showResults ? 'translate-x-0' : 'translate-x-full'
        } w-[28rem] overflow-y-auto`}
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
    </div>
  )
}

export default MapPage