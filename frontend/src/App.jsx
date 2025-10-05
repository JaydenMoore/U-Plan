import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './components/HomePage'
import MapPage from './components/MapPage'
import RisingIssuesPage from './components/RisingIssuesPage'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/map" element={<MapPage />} />
        <Route path="/rising-issues" element={<RisingIssuesPage />} /> 
      </Routes>
    </Router>
  )
}

export default App