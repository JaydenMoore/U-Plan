import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const RisingIssuesPage = () => {
  const [issues, setIssues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchIssues = async () => {
      try {
        setLoading(true);
        // Ensure you are using the correct backend URL and port
        const response = await fetch('http://localhost:8001/rising-issues');
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch rising issues.');
        }
        const data = await response.json();
        setIssues(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchIssues();
  }, []);

  const IssueCard = ({ issue }) => (
    <div className="bg-white shadow-lg rounded-lg p-6 border border-gray-200 hover:shadow-xl transition-shadow duration-300">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-xl font-bold text-gray-800">{issue.issue_type}</h3>
          <p className="text-sm text-gray-500">
            {issue.avg_latitude.toFixed(4)}, {issue.avg_longitude.toFixed(4)}
          </p>
        </div>
        <div className="text-right">
            <div className="text-lg font-semibold text-blue-600">{issue.count} Reports</div>
            <p className="text-xs text-gray-500">in area</p>
        </div>
      </div>
      <div className="mt-4 space-y-3">
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600">Avg. Risk Score:</span>
          <span className="font-semibold text-white px-2 py-0.5 rounded-full bg-yellow-500">
            {issue.overall_risk_score_avg ? issue.overall_risk_score_avg.toFixed(1) : 'N/A'}/10
          </span>
        </div>
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600">Last Reported:</span>
          <span className="font-semibold text-gray-700">{new Date(issue.last_reported_at).toLocaleDateString()}</span>
        </div>
      </div>
      {issue.partner_ngo_name && issue.call_to_action_link && (
        <div className="mt-4 pt-4 border-t">
            <p className="text-xs text-gray-500 mb-2">Partner Organization:</p>
            <a href={issue.call_to_action_link} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-semibold">
                {issue.partner_ngo_name}
            </a>
        </div>
      )}
    </div>
  );

  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="absolute top-4 left-4 z-20">
        <Link 
          to="/" 
          className="inline-flex items-center px-4 py-2 bg-black bg-opacity-70 text-white text-sm font-medium rounded-full hover:bg-opacity-90 transition-all duration-300 shadow-lg border border-white border-opacity-20"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          Back to Home
        </Link>
      </div>

      <div className="container mx-auto px-4 py-24">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-4">Rising Community Issues</h1>
        <p className="text-center text-gray-600 mb-12">Aggregated from user feedback to identify emerging environmental concerns.</p>

        {loading && (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        )}
        {error && (
          <div className="text-center py-12">
            <p className="text-red-500 font-semibold">⚠️ Error</p>
            <p className="text-gray-600">{error}</p>
          </div>
        )}
        {!loading && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {issues.length > 0 ? (
              issues.map(issue => <IssueCard key={issue.id} issue={issue} />)
            ) : (
              <p className="col-span-full text-center text-gray-500">No rising issues found based on recent feedback.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RisingIssuesPage;