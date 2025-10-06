import React, { useRef, useEffect, useState, useCallback, forwardRef, useImperativeHandle } from 'react'
import { MapContainer, TileLayer, useMapEvents, Marker, Popup, Rectangle } from 'react-leaflet'
import L from 'leaflet'

// Fix default marker icons in Leaflet
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

let DefaultIcon = L.icon({
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

L.Marker.prototype.options.icon = DefaultIcon

// Custom marker icon (blue)
const blueIcon = L.icon({
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [30, 49],
  iconAnchor: [15, 49],
  popupAnchor: [1, -34],
  shadowSize: [49, 49],
  className: 'blue-marker'
})

// Component to handle map clicks and zoom changes
function MapClickHandler({ onLocationClick, onZoomChange, onZoomEnd, isLoadingAssessment }) {
  const map = useMapEvents({
    click: (e) => {
      if (isLoadingAssessment) {
        console.log('MapClickHandler: Click ignored - assessment in progress')
        return
      }
      const { lat, lng } = e.latlng
      console.log('MapClickHandler: Map clicked at', lat, lng)
      onLocationClick(lat, lng)
    },
    dblclick: (e) => {
      if (isLoadingAssessment) {
        console.log('MapClickHandler: Double-click ignored - assessment in progress')
        return
      }
      const { lat, lng } = e.latlng
      console.log('MapClickHandler: Map double-clicked at', lat, lng, '(forcing fresh assessment)')
      onLocationClick(lat, lng)
    },
    zoomend: (e) => {
      const zoom = e.target.getZoom()
      console.log('MapClickHandler: Zoom changed to', zoom)
      onZoomChange(zoom)
      onZoomEnd && onZoomEnd(zoom, e.target.getCenter())
    },
  })
  return null
}

// Debounce utility function
function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

const Map = forwardRef(({ onLocationClick, assessment, onToggleResults }, ref) => {
  const mapRef = useRef(null)
  const [markerPosition, setMarkerPosition] = useState(null)
  const [showPopup, setShowPopup] = useState(false)
  const [popupData, setPopupData] = useState(null) // For grid-specific data
  const [selectedGrid, setSelectedGrid] = useState(null) // For selected grid square
  const [showGridInfo, setShowGridInfo] = useState(false) // For grid info panel
  const [showInstructions, setShowInstructions] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [showSearch, setShowSearch] = useState(false)
  const [regionalRiskData, setRegionalRiskData] = useState([])
  const [showRegionalRisk, setShowRegionalRisk] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [currentZoom, setCurrentZoom] = useState(4)
  const [isLoadingRisk, setIsLoadingRisk] = useState(false)
  const [showLegend, setShowLegend] = useState(true)
  const [lastBounds, setLastBounds] = useState(null)
  const [showToolsPanel, setShowToolsPanel] = useState(false)
  const [isLoadingAssessment, setIsLoadingAssessment] = useState(false)
  const [assessmentCancelled, setAssessmentCancelled] = useState(false)
  const [pendingAssessment, setPendingAssessment] = useState(null)

  // Handle zoom changes - only update zoom level, don't regenerate grid automatically
  const handleZoomEnd = useCallback((newZoom, center) => {
    // Just update the zoom level, user can manually regenerate if needed
    console.log('Zoom changed to:', newZoom)
  }, [])

  // Debounced search function to reduce API calls
  const debouncedSearch = useCallback(
    debounce(async (query) => {
      if (!query.trim()) {
        setSearchResults([])
        setIsSearching(false)
        return
      }

      setIsSearching(true)
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=1`
        )
        const data = await response.json()
        setSearchResults(data)
      } catch (error) {
        console.error('Error searching location:', error)
        setSearchResults([])
      } finally {
        setIsSearching(false)
      }
    }, 300),
    []
  )

  // Generate circular grid coordinates based on center point and radius
  const generateCircularGrid = (centerLat, centerLng, radiusKm, gridSizeKm) => {
    const coordinates = []
    const latDegreeKm = 111 // Rough conversion
    const lngDegreeKm = 111 * Math.cos(centerLat * Math.PI / 180)
    
    const radiusLat = radiusKm / latDegreeKm
    const radiusLng = radiusKm / lngDegreeKm
    const gridSizeLat = gridSizeKm / latDegreeKm
    const gridSizeLng = gridSizeKm / lngDegreeKm
    
    // Generate grid points in a circular pattern
    for (let lat = centerLat - radiusLat; lat <= centerLat + radiusLat; lat += gridSizeLat) {
      for (let lng = centerLng - radiusLng; lng <= centerLng + radiusLng; lng += gridSizeLng) {
        // Check if point is within circular radius
        const distance = Math.sqrt(
          Math.pow((lat - centerLat) * latDegreeKm, 2) + 
          Math.pow((lng - centerLng) * lngDegreeKm, 2)
        )
        
        if (distance <= radiusKm) {
          coordinates.push({ 
            lat: lat, 
            lng: lng, 
            gridSize: gridSizeKm / latDegreeKm // Convert back to degrees for visualization
          })
        }
      }
    }
    
    return coordinates
  }

  // Function to generate zoom-responsive circular regional risk assessment
  const generateRegionalRisk = async (bounds) => {
    console.log('generateRegionalRisk called with bounds:', bounds)
    console.log('Current zoom level:', currentZoom)
    
    // Don't generate if assessment is cancelled
    if (assessmentCancelled) {
      console.log('Regional risk generation cancelled')
      return
    }
    
    setIsLoadingRisk(true)
    
    try {
      const centerLat = (bounds.north + bounds.south) / 2
      const centerLng = (bounds.east + bounds.west) / 2
      
      // Use strategic sampling with very few points for speed
      let sampleStrategy, radiusKm, gridDensity
      
      if (currentZoom >= 16) {
        sampleStrategy = 'dense_grid'  // Dense grid for lot-level analysis
        radiusKm = 0.5    // 500m radius
        gridDensity = 9   // 3x3 grid
      } else if (currentZoom >= 14) {
        sampleStrategy = 'medium_grid'
        radiusKm = 1      // 1km radius
        gridDensity = 9   // 3x3 grid
      } else if (currentZoom >= 12) {
        sampleStrategy = 'center_plus_8'
        radiusKm = 2      // 2km radius
        gridDensity = 9   // 3x3 grid
      } else if (currentZoom >= 10) {
        sampleStrategy = 'center_plus_8'  // Center + 8 directions
        radiusKm = 5      // 5km radius
        gridDensity = 4   // 2x2 grid
      } else {
        sampleStrategy = 'center_plus_4'
        radiusKm = 10     // 10km radius
        gridDensity = 4   // 2x2 grid
      }
      
      // Generate strategic sample points
      let coordinates = []
      
      if (sampleStrategy === 'dense_grid' || sampleStrategy === 'medium_grid') {
        // Generate a proper grid for higher zoom levels
        const latDegreeKm = 111
        const lngDegreeKm = 111 * Math.cos(centerLat * Math.PI / 180)
        const radiusLat = radiusKm / latDegreeKm
        const radiusLng = radiusKm / lngDegreeKm
        
        const gridSize = Math.sqrt(gridDensity) // 3 for 9 points, 2 for 4 points
        const stepLat = (radiusLat * 2) / (gridSize - 1)
        const stepLng = (radiusLng * 2) / (gridSize - 1)
        
        for (let i = 0; i < gridSize; i++) {
          for (let j = 0; j < gridSize; j++) {
            const lat = centerLat - radiusLat + (i * stepLat)
            const lng = centerLng - radiusLng + (j * stepLng)
            
            // Check if point is within circular radius
            const distance = Math.sqrt(
              Math.pow((lat - centerLat) * latDegreeKm, 2) + 
              Math.pow((lng - centerLng) * lngDegreeKm, 2)
            )
            
            if (distance <= radiusKm) {
              coordinates.push({ lat, lng, gridSize: 0.01 })
            }
          }
        }
      } else if (sampleStrategy === 'center_plus_8') {
        // 9 points: center + 8 directions (N, NE, E, SE, S, SW, W, NW)
        const latDegreeKm = 111
        const lngDegreeKm = 111 * Math.cos(centerLat * Math.PI / 180)
        const radiusLat = radiusKm / latDegreeKm
        const radiusLng = radiusKm / lngDegreeKm
        const halfRadiusLat = radiusLat * 0.7
        const halfRadiusLng = radiusLng * 0.7
        
        coordinates = [
          { lat: centerLat, lng: centerLng, gridSize: 0.01 }, // Center
          { lat: centerLat + halfRadiusLat, lng: centerLng, gridSize: 0.01 }, // N
          { lat: centerLat + halfRadiusLat, lng: centerLng + halfRadiusLng, gridSize: 0.01 }, // NE
          { lat: centerLat, lng: centerLng + halfRadiusLng, gridSize: 0.01 }, // E
          { lat: centerLat - halfRadiusLat, lng: centerLng + halfRadiusLng, gridSize: 0.01 }, // SE
          { lat: centerLat - halfRadiusLat, lng: centerLng, gridSize: 0.01 }, // S
          { lat: centerLat - halfRadiusLat, lng: centerLng - halfRadiusLng, gridSize: 0.01 }, // SW
          { lat: centerLat, lng: centerLng - halfRadiusLng, gridSize: 0.01 }, // W
          { lat: centerLat + halfRadiusLat, lng: centerLng - halfRadiusLng, gridSize: 0.01 }  // NW
        ]
      } else {
        // center_plus_4: Just 5 points: center + north/south/east/west
        const latDegreeKm = 111
        const lngDegreeKm = 111 * Math.cos(centerLat * Math.PI / 180)
        const radiusLat = radiusKm / latDegreeKm
        const radiusLng = radiusKm / lngDegreeKm
        
        coordinates = [
          { lat: centerLat, lng: centerLng, gridSize: 0.01 }, // Center
          { lat: centerLat + radiusLat, lng: centerLng, gridSize: 0.01 }, // North
          { lat: centerLat - radiusLat, lng: centerLng, gridSize: 0.01 }, // South
          { lat: centerLat, lng: centerLng + radiusLng, gridSize: 0.01 }, // East
          { lat: centerLat, lng: centerLng - radiusLng, gridSize: 0.01 }  // West
        ]
      }
      
      console.log(`Generated ${coordinates.length} strategic sample points (${sampleStrategy}, radius: ${radiusKm}km) for zoom level ${currentZoom}`)

      if (coordinates.length === 0) {
        throw new Error('No coordinates generated for the selected area')
      }

      // Check if cancelled before making API call
      if (assessmentCancelled) {
        console.log('Regional risk generation cancelled before API call')
        return
      }

      // Create timeout promise
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout')), 30000) // 30 seconds timeout
      })

      // Create fetch promise
      const fetchPromise = fetch('http://localhost:8001/assess-risk-bulk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ coordinates }),
      })

      // Race between fetch and timeout
      const response = await Promise.race([fetchPromise, timeoutPromise])
      
      // Check if cancelled after API call
      if (assessmentCancelled) {
        console.log('Regional risk generation cancelled after API call')
        return
      }
      
      console.log('Bulk API response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Bulk API error response:', errorText)
        throw new Error(`Failed to fetch regional risk data: ${response.status} - ${errorText}`)
      }

      const data = await response.json()
      
      if (!data.results || data.results.length === 0) {
        throw new Error('No risk assessment data received from API')
      }

      // Create interpolated visualization from strategic sample points
      const validResults = data.results.filter((result, index) => {
        return result && result.flood_risk && result.heat_risk
      })
      
      if (validResults.length === 0) {
        throw new Error('No valid risk assessment data received')
      }
      
      // Calculate dominant risk levels from sample points
      const floodRisks = validResults.map(r => r.flood_risk)
      const heatRisks = validResults.map(r => r.heat_risk)
      
      // Find most common risk level (simple mode)
      const getMode = (arr) => {
        const counts = {}
        arr.forEach(item => counts[item] = (counts[item] || 0) + 1)
        return Object.entries(counts).reduce((a, b) => counts[a[0]] > counts[b[0]] ? a : b)[0]
      }
      
      const dominantFloodRisk = getMode(floodRisks)
      const dominantHeatRisk = getMode(heatRisks)
      
      // Create visualization rectangles based on grid density
      const latDegreeKm = 111
      const lngDegreeKm = 111 * Math.cos(centerLat * Math.PI / 180)
      const radiusLat = radiusKm / latDegreeKm
      const radiusLng = radiusKm / lngDegreeKm
      
      const riskRectangles = []
      
      // Create grid rectangles based on zoom level and strategy
      const gridDivisions = Math.sqrt(gridDensity) // 2 for 4 squares, 3 for 9 squares
      const rectSizeLat = (radiusLat * 2) / gridDivisions
      const rectSizeLng = (radiusLng * 2) / gridDivisions
      
      for (let i = 0; i < gridDivisions; i++) {
        for (let j = 0; j < gridDivisions; j++) {
          const rectLat = centerLat - radiusLat + (i + 0.5) * rectSizeLat
          const rectLng = centerLng - radiusLng + (j + 0.5) * rectSizeLng
          
          // Check if rectangle center is within circular radius
          const distance = Math.sqrt(
            Math.pow((rectLat - centerLat) * latDegreeKm, 2) + 
            Math.pow((rectLng - centerLng) * lngDegreeKm, 2)
          )
          
          if (distance <= radiusKm) {
            // Vary the risk slightly for visual interest while keeping dominant pattern
            let localFloodRisk = dominantFloodRisk
            let localHeatRisk = dominantHeatRisk
            
            // Add some variation based on sample data if available
            if (validResults.length > (i * gridDivisions + j)) {
              const sampleResult = validResults[i * gridDivisions + j]
              if (sampleResult) {
                localFloodRisk = sampleResult.flood_risk
                localHeatRisk = sampleResult.heat_risk
              }
            }
            
            riskRectangles.push({
              id: `risk-grid-${i}-${j}`,
              bounds: [
                [rectLat - rectSizeLat/2, rectLng - rectSizeLng/2],
                [rectLat + rectSizeLat/2, rectLng + rectSizeLng/2]
              ],
              risk: localFloodRisk,
              heatRisk: localHeatRisk,
              color: getRiskColor(localFloodRisk),
              borderColor: getRiskColor(localHeatRisk),
              opacity: getRiskOpacity(localFloodRisk) * 0.8,
              assessmentType: sampleStrategy
            })
          }
        }
      }

      // Only update if not cancelled
      if (!assessmentCancelled) {
        setRegionalRiskData(riskRectangles)
        setLastBounds(bounds)
        console.log(`Created ${riskRectangles.length} risk grid rectangles from ${coordinates.length} sample points`)
        
        console.log(`Regional Assessment Complete: ${coordinates.length} sample points → ${riskRectangles.length} grid areas in ${radiusKm}km radius`)
        console.log(`Strategy: ${sampleStrategy}, Dominant risks - Flood: ${dominantFloodRisk}, Heat: ${dominantHeatRisk}`)
      }

    } catch (error) {
      console.error('Error generating regional risk:', error)
      let errorMessage = error.message || 'Unknown error occurred'
      
      if (!assessmentCancelled) {
        if (error.message === 'Request timeout') {
          alert('Request timeout: The NASA API is taking too long to respond. Please try with a smaller area or wait a moment and try again.')
        } else if (error.message.includes('Failed to fetch')) {
          alert('Network error: Could not connect to the backend server. Please ensure the backend is running on port 8001.')
        } else {
          alert('Error generating regional risk assessment: ' + errorMessage)
        }
      }
    } finally {
      if (!assessmentCancelled) {
        setIsLoadingRisk(false)
      }
    }
  }

  // Handle search result selection
  const handleSearchSelect = (result) => {
    const lat = parseFloat(result.lat)
    const lng = parseFloat(result.lon)
    
    setSearchQuery(result.display_name.split(',')[0]) // Set to city/location name
    setSearchResults([])
    setShowSearch(false)
    
    // Set marker position to show the searched location
    setMarkerPosition([lat, lng])
    setShowPopup(true)
    
    // Determine appropriate zoom level based on result type
    let targetZoom = 15 // Default for cities
    if (result.type === 'house' || result.type === 'building') {
      targetZoom = 18 // Zoom very close for buildings
    } else if (result.type === 'way' || result.type === 'residential') {
      targetZoom = 17 // Zoom close for streets/residential areas
    } else if (result.class === 'place' && (result.type === 'village' || result.type === 'town')) {
      targetZoom = 14 // Moderate zoom for villages/towns
    } else if (result.class === 'place' && result.type === 'city') {
      targetZoom = 12 // Wider view for cities
    }
    
    // Update map view
    if (mapRef.current) {
      const map = mapRef.current
      map.setView([lat, lng], targetZoom, { animate: true, duration: 1 })
      setCurrentZoom(targetZoom) // Update zoom state
    }
    
    // Generate regional risk for the area with zoom-appropriate bounds
    const boundsSize = targetZoom >= 10 ? 0.1 : (targetZoom >= 8 ? 0.25 : 0.5)
    const bounds = {
      north: lat + boundsSize,
      south: lat - boundsSize,
      east: lng + boundsSize,
      west: lng - boundsSize
    }
    
    // Always store bounds for future use
    setLastBounds(bounds)
    
    // Generate regional risk if enabled
    if (showRegionalRisk) {
      generateRegionalRisk(bounds)
    }
    
    onLocationClick(lat, lng)
  }

  // Get color for risk level
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High': return '#ef4444' // red
      case 'Medium': return '#f59e0b' // amber
      case 'Low': return '#10b981' // green
      default: return '#6b7280' // gray
    }
  }

  // Get opacity for risk level
  const getRiskOpacity = (risk) => {
    switch (risk) {
      case 'High': return 0.6
      case 'Medium': return 0.4
      case 'Low': return 0.2
      default: return 0.1
    }
  }

  // Handle clicking on regional risk grid squares
  const handleGridSquareClick = (gridData, lat, lng) => {
    console.log('Grid square clicked:', lat, lng, 'Grid data:', gridData)
    
    // Update popup to show grid-specific data
    setPopupData({
      isGridSquare: true,
      gridFloodRisk: gridData.risk,
      gridHeatRisk: gridData.heatRisk,
      gridColor: gridData.color,
      gridBorderColor: gridData.borderColor,
      assessmentType: gridData.assessmentType || 'interpolated'
    })
    
    // Auto-hide popup after 5 seconds (longer for grid data)
    setTimeout(() => {
      setPopupData(null)
    }, 5000)
  }

  const handleLocationClick = (lat, lng) => {
    console.log('Map click handled:', lat, lng, 'Regional risk enabled:', showRegionalRisk)
    
    // Prevent new clicks if already loading an assessment
    if (isLoadingAssessment) {
      console.log('Assessment already in progress, ignoring click')
      return
    }
    
    // Start loading assessment
    setIsLoadingAssessment(true)
    setAssessmentCancelled(false)
    setPendingAssessment({ lat, lng })
    
    // Generate bounds for regional risk based on current zoom
    const boundsSize = currentZoom >= 10 ? 0.1 : (currentZoom >= 8 ? 0.25 : 0.5)
    const bounds = {
      north: lat + boundsSize,
      south: lat - boundsSize,
      east: lng + boundsSize,
      west: lng - boundsSize
    }
    
    // Clear grid selection when clicking elsewhere
    setSelectedGrid(null)
    setShowGridInfo(false)
    
    // Always update marker position and popup
    setMarkerPosition([lat, lng])
    setShowPopup(true)
    setPopupData(null) // Clear any grid-specific data
    
    // Always store bounds for future use
    setLastBounds(bounds)
    
    // If regional risk is enabled, clear existing grid and regenerate for new location
    if (showRegionalRisk) {
      console.log('Clearing existing grid and regenerating for new location')
      // Clear existing regional risk data
      setRegionalRiskData([])
      
      // Generate new regional risk for the new location
      generateRegionalRisk(bounds)
    }
    
    // Always call the parent handler to trigger individual assessment
    console.log('Calling parent onLocationClick handler')
    if (onLocationClick) {
      onLocationClick(lat, lng)
    }
  }

  const handleZoomChange = (zoom) => {
    setCurrentZoom(zoom)
  }

  const handleCancelAssessment = () => {
    console.log('Assessment cancelled by user')
    setIsLoadingAssessment(false)
    setAssessmentCancelled(true)
    setPendingAssessment(null)
    
    // Auto-hide popup after cancellation
    setTimeout(() => {
      setShowPopup(false)
      setAssessmentCancelled(false)
    }, 2000)
  }

  // Function to complete assessment (call this from parent component)
  const completeAssessment = useCallback(() => {
    console.log('Assessment completed')
    setIsLoadingAssessment(false)
    setPendingAssessment(null)
    
    // Auto-hide popup after assessment completes
    setTimeout(() => {
      setShowPopup(false)
    }, 5000)
  }, [])

  // Expose the completeAssessment function for parent component
  useImperativeHandle(ref, () => ({
    completeAssessment
  }), [completeAssessment])

  useEffect(() => {
    if (mapRef.current) {
      const map = mapRef.current
      map.on('zoom', (e) => {
        setCurrentZoom(e.target.getZoom())
      })
    }
    
    // Cleanup function
    return () => {
      // Any cleanup code can go here
    }
  }, [])

  return (
    <div className="relative h-full w-full">
      <MapContainer
        ref={mapRef}
        center={[39.8283, -98.5795]} // Center of US
        zoom={4}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg shadow-lg"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Regional Risk Grid */}
        {showRegionalRisk && regionalRiskData.length > 0 && console.log('Rendering', regionalRiskData.length, 'regional risk rectangles')}
        {showRegionalRisk && regionalRiskData.map((rect) => (
          <Rectangle
            key={rect.id}
            bounds={rect.bounds}
            pathOptions={{
              fillColor: rect.color,
              fillOpacity: selectedGrid?.id === rect.id ? rect.opacity + 0.3 : rect.opacity,
              color: selectedGrid?.id === rect.id ? '#ffffff' : rect.borderColor, // Border color represents heat risk
              weight: selectedGrid?.id === rect.id ? 4 : 2,
              opacity: selectedGrid?.id === rect.id ? 1 : 0.8
            }}
            eventHandlers={{
              click: (e) => {
                if (isLoadingAssessment) {
                  console.log('Grid click ignored - assessment in progress')
                  return
                }
                
                // Calculate center of the grid square
                const centerLat = (rect.bounds[0][0] + rect.bounds[1][0]) / 2
                const centerLng = (rect.bounds[0][1] + rect.bounds[1][1]) / 2
                
                console.log('Rectangle clicked, selecting grid square')
                
                // Set selected grid for highlighting
                setSelectedGrid(rect)
                setShowGridInfo(true)
                
                // Also call location handler for individual assessment
                handleLocationClick(centerLat, centerLng)
              }
            }}
          />
        ))}
        
        <MapClickHandler 
          onLocationClick={handleLocationClick} 
          onZoomChange={handleZoomChange}
          onZoomEnd={handleZoomEnd}
          isLoadingAssessment={isLoadingAssessment}
        />
        
        {markerPosition && (
          <Marker position={markerPosition} icon={blueIcon}>
            {showPopup && (
              <Popup autoClose={false} closeButton={false}>
                <div className="text-center p-2">
                  {popupData && popupData.isGridSquare ? (
                    // Grid square popup
                    <div>
                      <p className="text-sm font-medium text-gray-800 mb-1">🎯 Grid Square Details</p>
                      <p className="text-xs text-gray-600 mb-2">
                        {markerPosition[0].toFixed(4)}, {markerPosition[1].toFixed(4)}
                      </p>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span>🌊 Flood Risk:</span>
                          <span className={`font-medium px-2 py-0.5 rounded text-white`} 
                                style={{backgroundColor: popupData.gridColor}}>
                            {popupData.gridFloodRisk}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span>🔥 Heat Risk:</span>
                          <span className={`font-medium px-2 py-0.5 rounded text-white`} 
                                style={{backgroundColor: popupData.gridBorderColor}}>
                            {popupData.gridHeatRisk}
                          </span>
                        </div>
                        {popupData.airQualityData && (
                          <>
                            <div className="flex items-center justify-between text-xs">
                              <span>🏭 Air Quality:</span>
                              <span className={`font-medium px-2 py-0.5 rounded text-white ${
                                popupData.airQualityData.air_quality_risk === 'High' ? 'bg-red-600' :
                                popupData.airQualityData.air_quality_risk === 'Medium' ? 'bg-yellow-600' :
                                popupData.airQualityData.air_quality_risk === 'Low' ? 'bg-green-600' : 'bg-blue-600'
                              }`}>
                                {popupData.airQualityData.air_quality_risk}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span>📊 AQI:</span>
                              <span className="font-medium text-gray-700">
                                {popupData.airQualityData.air_quality_index}/5
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span>💨 PM2.5:</span>
                              <span className="font-medium text-gray-700">
                                {popupData.airQualityData.pm2_5.toFixed(2)} μg/m³
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span>🏆 Overall Risk:</span>
                              <span className={`font-medium px-2 py-0.5 rounded text-white ${
                                popupData.airQualityData.overall_risk_score >= 7 ? 'bg-red-600' :
                                popupData.airQualityData.overall_risk_score >= 5 ? 'bg-yellow-600' :
                                popupData.airQualityData.overall_risk_score >= 3 ? 'bg-green-600' : 'bg-blue-600'
                              }`}>
                                {popupData.airQualityData.overall_risk_score}/10
                              </span>
                            </div>
                          </>
                        )}
                        <p className="text-xs text-gray-500 mt-1">
                          Assessment: {popupData.assessmentType}
                        </p>
                      </div>
                    </div>
                  ) : isLoadingAssessment ? (
                    // Loading state popup
                    <div>
                      <p className="text-sm font-medium text-gray-800 mb-2">� Analyzing Location</p>
                      <p className="text-xs text-gray-600 mb-3">
                        {markerPosition[0].toFixed(4)}, {markerPosition[1].toFixed(4)}
                      </p>
                      <div className="flex items-center justify-center mb-3">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                      </div>
                      <p className="text-xs text-blue-600 mb-3">
                        Assessing climate risks from NASA data...
                      </p>
                      <button
                        onClick={handleCancelAssessment}
                        className="bg-red-500 hover:bg-red-600 text-white text-xs px-3 py-1 rounded transition-colors duration-200"
                      >
                        Cancel Assessment
                      </button>
                    </div>
                  ) : assessmentCancelled ? (
                    // Cancelled state popup
                    <div>
                      <p className="text-sm font-medium text-gray-800 mb-1">❌ Assessment Cancelled</p>
                      <p className="text-xs text-gray-600 mb-2">
                        {markerPosition[0].toFixed(4)}, {markerPosition[1].toFixed(4)}
                      </p>
                      <p className="text-xs text-red-600">
                        Click elsewhere to start a new assessment
                      </p>
                    </div>
                  ) : (
                    // Regular location popup (completed assessment)
                    <div>
                      <p className="text-sm font-medium text-gray-800 mb-1">📍 Assessment Complete</p>
                      <p className="text-xs text-gray-600 mb-2">
                        {markerPosition[0].toFixed(4)}, {markerPosition[1].toFixed(4)}
                      </p>
                      <p className="text-xs text-green-600">
                        Check results panel for detailed analysis
                      </p>
                    </div>
                  )}
                </div>
              </Popup>
            )}
          </Marker>
        )}
        

      </MapContainer>

      {/* Instructions Overlay */}
      {showInstructions && (
        <div className="absolute top-4 left-4 bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-4 shadow-lg z-[1000] border border-gray-200 max-w-sm">
          <div className="flex items-start justify-between mb-2">
            <div>
              <p className="text-sm font-medium text-gray-800">🏙️ Urban Planning Tools:</p>
            </div>
            <button
              onClick={() => setShowInstructions(false)}
              className="text-gray-400 hover:text-gray-600 ml-2 text-lg leading-none"
              aria-label="Close instructions"
            >
              ×
            </button>
          </div>
          <div className="text-xs text-gray-600 space-y-1">
            <p>• Click anywhere on the map to assess climate risks</p>
            <p>• Use 🚨 button to show/hide risk assessment panel</p>
            <p>• Use ❓ button to show/hide user guide</p>
            <p>• Use 🔍 Search to find specific locations</p>
            <p>• Use 🛠️ Tools button to access regional grid and options</p>
            <p>• Use 📊 button to show/hide planning legend</p>
            <p>• Zoom in for more detailed lot-level analysis</p>
          </div>
        </div>
      )}

      {/* Bottom Left Controls Panel */}
      <div className="absolute bottom-4 left-2 z-[1001] flex flex-col space-y-2">
        {/* View Results Button */}
        <button
          onClick={() => {
            if (onToggleResults) {
              onToggleResults()
            }
          }}
          className="bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-2 shadow-lg border border-gray-200 hover:bg-opacity-100 transition-all duration-200"
          aria-label="View Results"
        >
          🚨
        </button>
        
        {/* User Guide Button */}
        <button
          onClick={() => setShowInstructions(!showInstructions)}
          className="bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-2 shadow-lg border border-gray-200 hover:bg-opacity-100 transition-all duration-200"
          aria-label="Toggle user guide"
        >
          ❓
        </button>
        
        {/* Search Toggle Button */}
        <button
          onClick={() => setShowSearch(!showSearch)}
          className="bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-2 shadow-lg border border-gray-200 hover:bg-opacity-100 transition-all duration-200"
          aria-label="Search locations"
        >
          🔍
        </button>
        
        {/* Tools Panel Toggle */}
        <button
          onClick={() => setShowToolsPanel(!showToolsPanel)}
          className="bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-2 shadow-lg border border-gray-200 hover:bg-opacity-100 transition-all duration-200"
          aria-label="Toggle tools panel"
        >
          🛠️
        </button>
        
        {/* Legend Toggle Button */}
        <button
          onClick={() => setShowLegend(!showLegend)}
          className="bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-2 shadow-lg border border-gray-200 hover:bg-opacity-100 transition-all duration-200"
          aria-label={showLegend ? "Hide legend" : "Show legend"}
        >
          📊
        </button>
      </div>
      
      {/* Search Interface Overlay */}
      {showSearch && (
        <div className="absolute bottom-16 left-2 z-[1002] bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-3 shadow-lg border border-gray-200 max-w-sm">
          <div className="flex items-start justify-between mb-2">
            <p className="text-sm font-medium text-gray-800">🔍 Location Search</p>
            <button
              onClick={() => setShowSearch(false)}
              className="text-gray-400 hover:text-gray-600 ml-2 text-lg leading-none"
              aria-label="Close search"
            >
              ×
            </button>
          </div>
          
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                debouncedSearch(e.target.value)
              }}
              placeholder="Search for a location..."
              className="w-full px-3 py-2 pr-8 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            />
            {isSearching && (
              <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              </div>
            )}
          </div>
          
          {searchResults.length > 0 && (
            <div className="mt-2 bg-white border border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
              {searchResults.map((result, index) => (
                <button
                  key={index}
                  onClick={() => handleSearchSelect(result)}
                  className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 border-b border-gray-100 last:border-b-0"
                >
                  <div className="font-medium text-gray-800">{result.display_name.split(',')[0]}</div>
                  <div className="text-xs text-gray-500">{result.display_name}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Grid Info Panel */}
      {showGridInfo && selectedGrid && (
        <div className="absolute top-4 right-4 z-[1003] bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-4 shadow-lg border border-gray-200 max-w-sm">
          <div className="flex items-start justify-between mb-3">
            <p className="text-sm font-medium text-gray-800">📊 Grid Square Analysis</p>
            <button
              onClick={() => {
                setShowGridInfo(false)
                setSelectedGrid(null)
              }}
              className="text-gray-400 hover:text-gray-600 ml-2 text-lg leading-none"
              aria-label="Close grid info"
            >
              ×
            </button>
          </div>
          
          <div className="space-y-3">
            <div className="bg-blue-50 rounded-lg p-3">
              <p className="text-xs text-blue-600 mb-1">Coordinates</p>
              <p className="text-sm font-mono text-blue-800">
                {((selectedGrid.bounds[0][0] + selectedGrid.bounds[1][0]) / 2).toFixed(4)}, {((selectedGrid.bounds[0][1] + selectedGrid.bounds[1][1]) / 2).toFixed(4)}
              </p>
              <p className="text-xs text-blue-600 mt-1">
                Area: {(Math.abs((selectedGrid.bounds[1][0] - selectedGrid.bounds[0][0]) * (selectedGrid.bounds[1][1] - selectedGrid.bounds[0][1])) * 111 * 111).toFixed(2)} km²
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="flex items-center text-sm">
                  <span className="w-3 h-3 rounded mr-2" style={{backgroundColor: selectedGrid.color}}></span>
                  🌊 Flood Risk:
                </span>
                <span className="font-semibold text-blue-700">{selectedGrid.risk}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="flex items-center text-sm">
                  <span className="w-3 h-3 rounded mr-2" style={{backgroundColor: selectedGrid.borderColor}}></span>
                  🔥 Heat Risk:
                </span>
                <span className="font-semibold text-red-700">{selectedGrid.heatRisk}</span>
              </div>
            </div>
            
            <div className="border-t pt-2">
              <p className="text-xs text-gray-500">
                Assessment: {selectedGrid.assessmentType || 'interpolated'}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Click elsewhere on the map to deselect
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tools Panel */}
      {showToolsPanel && (
        <div className="absolute bottom-16 left-20 z-[1002] bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-3 shadow-lg border border-gray-200 max-w-sm">
          <div className="flex items-start justify-between mb-2">
            <p className="text-sm font-medium text-gray-800">🛠️ Planning Tools</p>
            <button
              onClick={() => setShowToolsPanel(false)}
              className="text-gray-400 hover:text-gray-600 ml-2 text-lg leading-none"
              aria-label="Close tools panel"
            >
              ×
            </button>
          </div>
          
          <div className="space-y-3">
            {/* Regional Risk Grid Toggle */}
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="regionalRiskToggle"
                checked={showRegionalRisk}
                onChange={(e) => {
                  console.log('Regional risk checkbox toggled:', e.target.checked)
                  setShowRegionalRisk(e.target.checked)
                  if (!e.target.checked) {
                    // Clear existing regional risk data when hiding
                    setRegionalRiskData([])
                  } else {
                    // If we have stored bounds, regenerate grid
                    if (lastBounds) {
                      console.log('Regenerating grid with stored bounds:', lastBounds)
                      generateRegionalRisk(lastBounds)
                    } else if (markerPosition) {
                      // If we have a marker but no bounds, create bounds from marker position
                      console.log('Creating bounds from marker position:', markerPosition)
                      const [lat, lng] = markerPosition
                      const boundsSize = currentZoom >= 10 ? 0.1 : (currentZoom >= 8 ? 0.25 : 0.5)
                      const bounds = {
                        north: lat + boundsSize,
                        south: lat - boundsSize,
                        east: lng + boundsSize,
                        west: lng - boundsSize
                      }
                      setLastBounds(bounds)
                      generateRegionalRisk(bounds)
                    } else {
                      console.log('No location selected yet - grid will appear when location is clicked')
                    }
                    // If no location is selected yet, the grid will appear when they click somewhere
                  }
                }}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
              />
              <label htmlFor="regionalRiskToggle" className="text-sm text-gray-700 cursor-pointer">
                Show Regional Risk Grid
              </label>
            </div>
            
            {showRegionalRisk && (
              <div className="ml-7 text-xs text-gray-500 space-y-1">
                <p>Grid adapts to zoom level:</p>
                <p>• Zoom 18+: Individual lots (~50m)</p>
                <p>• Zoom 16-17: City blocks (~100m)</p>
                <p>• Zoom 14-15: Neighborhoods (~200m)</p>
                <p>• Zoom 8-13: Larger areas (~500m-5km)</p>
                <p className="text-blue-600 font-medium mt-2">💡 Click any grid square for detailed stats!</p>
              </div>
            )}
            
            {/* Loading indicator for regional risk */}
            {isLoadingRisk && (
              <div className="flex items-center space-x-2 text-blue-600">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-xs">Generating risk assessment...</span>
              </div>
            )}
            
            {/* Future tools can be added here */}
            <div className="pt-2 border-t border-gray-200">
              <p className="text-xs text-gray-500">
                More planning tools coming soon...
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Collapsible Urban Planning Legend */}
      {showLegend && (
        <div className="absolute top-4 right-4 bg-white bg-opacity-95 backdrop-blur-sm rounded-lg p-3 shadow-lg z-[1000] border border-gray-200 max-w-xs">
          <div className="flex items-start justify-between mb-2">
            <p className="text-xs font-medium text-gray-800">🏙️ Urban Planning Risk Assessment</p>
            <button
              onClick={() => setShowLegend(false)}
              className="text-gray-400 hover:text-gray-600 ml-2 text-lg leading-none"
              aria-label="Close legend"
            >
              ×
            </button>
          </div>
          
          {/* Current Zoom Scale */}
          <div className="mb-3 p-2 bg-blue-50 rounded">
            <p className="text-xs font-medium text-blue-800 mb-1">
              Current Scale: 
              {currentZoom >= 18 ? ' 🏠 Individual Lot (~50m)' :
               currentZoom >= 16 ? ' 🏘️ City Block (~100m)' :
               currentZoom >= 14 ? ' 🏙️ Small Neighborhood (~200m)' :
               currentZoom >= 12 ? ' 🏢 Large Block (~500m)' :
               currentZoom >= 10 ? ' 🌆 District (~1km)' :
               currentZoom >= 8 ? ' 🗺️ Neighborhood (~5km)' :
               ' 🌍 Regional (~25km)'}
            </p>
            {currentZoom >= 16 && (
              <p className="text-xs text-blue-600">Perfect for development site analysis</p>
            )}
            {currentZoom >= 12 && currentZoom < 16 && (
              <p className="text-xs text-blue-600">Ideal for zoning and block planning</p>
            )}
          </div>
          
          {/* Risk Markers */}
          <div className="mb-3">
            <p className="text-xs font-medium text-gray-700 mb-1">📍 Point Markers:</p>
            <div className="space-y-1 text-xs">
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                <span className="text-gray-700">Low Risk</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
                <span className="text-gray-700">Medium Risk</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                <span className="text-gray-700">High Risk</span>
              </div>
            </div>
          </div>
          
          {/* Regional Risk Areas */}
          {showRegionalRisk && (
            <div className="border-t pt-2">
              <p className="text-xs font-medium text-gray-700 mb-1">🗺️ Site Analysis Grid:</p>
              <div className="space-y-1 text-xs">
                <div className="flex items-center">
                  <div className="w-4 h-3 bg-green-500 bg-opacity-40 border border-green-600 mr-2"></div>
                  <span className="text-gray-700">Low Flood Risk</span>
                </div>
                <div className="flex items-center">
                  <div className="w-4 h-3 bg-yellow-500 bg-opacity-40 border border-yellow-600 mr-2"></div>
                  <span className="text-gray-700">Medium Flood Risk</span>
                </div>
                <div className="flex items-center">
                  <div className="w-4 h-3 bg-red-500 bg-opacity-60 border border-red-600 mr-2"></div>
                  <span className="text-gray-700">High Flood Risk</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Border color indicates heat risk level
              </p>
              <p className="text-xs text-blue-600 mt-1">
                Zoom in for finer site analysis
              </p>
            </div>
          )}
        </div>
      )}

      {/* Attribution */}
      <div className="absolute bottom-2 right-2 text-xs text-gray-600 bg-white bg-opacity-90 px-2 py-1 rounded shadow z-[1000]">
        © OpenStreetMap contributors
      </div>

      {/* Assessment Loading Overlay */}
      {isLoadingAssessment && (
        <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center z-[2000] pointer-events-none">
          <div className="bg-white rounded-lg p-6 shadow-xl flex items-center space-x-4 max-w-sm mx-4 pointer-events-auto">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <div>
              <p className="text-lg font-medium text-gray-800">Analyzing Climate Data</p>
              <p className="text-sm text-gray-600">Processing NASA satellite data...</p>
              <button
                onClick={handleCancelAssessment}
                className="mt-2 bg-red-500 hover:bg-red-600 text-white text-sm px-4 py-1 rounded transition-colors duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Regional Risk Loading Overlay */}
      {isLoadingRisk && (
        <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center z-[1500] pointer-events-none">
          <div className="bg-white rounded-lg p-4 shadow-xl flex items-center space-x-3 max-w-sm mx-4 pointer-events-auto">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <div>
              <p className="text-sm font-medium text-gray-800">Generating Regional Grid</p>
              <p className="text-xs text-gray-600">Analyzing area coverage...</p>
            </div>
          </div>
        </div>
      )}

      {/* Custom CSS for colored markers */}
      <style>{`
        .blue-marker {
          filter: hue-rotate(240deg) saturate(150%);
        }
        .red-marker {
          filter: hue-rotate(0deg) saturate(200%) brightness(0.8);
        }
        .yellow-marker {
          filter: hue-rotate(45deg) saturate(200%) brightness(1.2);
        }
        .green-marker {
          filter: hue-rotate(120deg) saturate(150%) brightness(1.1);
        }
      `}</style>
    </div>
  )
})

Map.displayName = 'Map'

export default Map