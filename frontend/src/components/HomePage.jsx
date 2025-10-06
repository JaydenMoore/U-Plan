import React from 'react'
import { Link } from 'react-router-dom'

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center">
            <div className="text-3xl mr-3">🌍</div>
            <h1 className="text-3xl font-bold text-gray-900">Urban Planner AI</h1>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-900 mb-6">
            NASA-Powered Urban Planning
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Harness the power of NASA's climate data to make informed urban planning decisions. 
            Assess environmental risks, analyze climate patterns, and plan sustainable communities 
            with precision and confidence.
          </p>
          <div className="flex justify-center gap-4">
            <Link
              to="/map"
              className="inline-flex items-center px-8 py-4 text-lg font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
            >
              🗺️ Launch Planning Tool
              <svg className="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link 
              to="/rising-issues" 
              className="inline-flex items-center px-8 py-4 text-lg font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
            >
              📈 Rising Issues
            </Link>
            <Link 
              to="/dashboard" 
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg shadow hover:shadow-md transition-all duration-300 border-2 border-gray-300"
              title="Developer Dashboard - System Monitoring"
            >
              🔧 Dev Dashboard
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">🌡️</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Climate Risk Assessment</h3>
            <p className="text-gray-600">
              Real-time analysis of temperature, precipitation, and extreme weather patterns using NASA POWER API data.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">🌊</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Flood Risk Analysis</h3>
            <p className="text-gray-600">
              Advanced flood risk modeling based on precipitation data, topography, and historical climate patterns.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">🏗️</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Site-Level Planning</h3>
            <p className="text-gray-600">
              Zoom from regional planning to individual lot analysis with adaptive grid systems for precise planning.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">📊</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Interactive Mapping</h3>
            <p className="text-gray-600">
              Dynamic maps with search functionality, regional risk grids, and comprehensive planning tools.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">🔄</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Real-Time Data</h3>
            <p className="text-gray-600">
              Access up-to-date environmental data directly from NASA's satellite observations and climate models.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="text-4xl mb-4">🎯</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Smart Recommendations</h3>
            <p className="text-gray-600">
              AI-powered suggestions for sustainable development based on environmental risk assessments.
            </p>
          </div>
        </div>

        {/* About Section */}
        <div className="bg-white rounded-xl p-8 shadow-lg mb-12">
          <h3 className="text-3xl font-bold text-gray-900 mb-6 text-center">About the Project</h3>
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <p className="text-gray-600 mb-4">
                Urban Planner AI was developed for the NASA Space Apps Challenge 2025, combining cutting-edge 
                space technology with urban planning expertise to address climate challenges in city development.
              </p>
              <p className="text-gray-600 mb-4">
                By leveraging NASA's POWER (Prediction of Worldwide Energy Resources) API, we provide urban 
                planners with unprecedented access to climate data for making informed decisions about 
                sustainable development.
              </p>
              <p className="text-gray-600">
                Our platform supports planning at multiple scales - from regional assessments covering 
                thousands of square kilometers to lot-level analysis for individual building sites.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white rounded-xl p-6">
                <div className="text-3xl mb-2">🚀</div>
                <h4 className="text-xl font-semibold mb-2">NASA Space Apps 2025</h4>
                <p className="text-blue-100">
                  Solving climate challenges through space-powered innovation
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="bg-gray-50 rounded-xl p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">Built With</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="flex flex-col items-center">
              <div className="bg-white rounded-lg p-4 shadow-md mb-2">
                <span className="text-2xl">🗺️</span>
              </div>
              <span className="text-sm font-medium text-gray-700">Leaflet Maps</span>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-white rounded-lg p-4 shadow-md mb-2">
                <span className="text-2xl">⚛️</span>
              </div>
              <span className="text-sm font-medium text-gray-700">React</span>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-white rounded-lg p-4 shadow-md mb-2">
                <span className="text-2xl">🐍</span>
              </div>
              <span className="text-sm font-medium text-gray-700">FastAPI</span>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-white rounded-lg p-4 shadow-md mb-2">
                <span className="text-2xl">🛰️</span>
              </div>
              <span className="text-sm font-medium text-gray-700">NASA POWER</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-sm mb-2">
            Built with ❤️ for NASA Space Apps Challenge 2025 | 
            Data powered by NASA POWER API
          </p>
          <a 
            href="https://github.com/JaydenMoore/U-Plan" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            View on GitHub
          </a>
        </div>
      </footer>
    </div>
  )
}

export default HomePage