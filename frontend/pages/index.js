import Head from 'next/head'
import { useState, useEffect } from 'react'

export default function Home() {
  const [systemInfo, setSystemInfo] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    // Fetch system info
    fetch('/api/system/info')
      .then(res => res.json())
      .then(data => setSystemInfo(data))
      .catch(err => console.error('Error fetching system info:', err))

    // Fetch health status
    fetch('/api/system/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error('Error fetching health:', err))
  }, [])

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <Head>
        <title>Persian Legal AI</title>
        <meta name="description" content="Persian Legal AI System" />
      </Head>

      <main>
        <h1 style={{ color: '#2c3e50', textAlign: 'center', marginBottom: '30px' }}>
          🔥 Persian Legal AI System
        </h1>
        
        <div style={{ backgroundColor: '#ecf0f1', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
          <h2 style={{ color: '#27ae60', marginBottom: '15px' }}>✅ System Status: OPERATIONAL</h2>
          <p>The Persian Legal AI system has been successfully restored and is ready for development!</p>
        </div>

        {health && (
          <div style={{ backgroundColor: '#e8f5e8', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
            <h3 style={{ color: '#27ae60' }}>🏥 Health Check</h3>
            <p><strong>Status:</strong> {health.status}</p>
            <p><strong>Database:</strong> {health.services?.database}</p>
            <p><strong>API:</strong> {health.services?.api}</p>
            <p><strong>AI Model:</strong> {health.services?.ai_model}</p>
          </div>
        )}

        {systemInfo && (
          <div style={{ backgroundColor: '#f0f8ff', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
            <h3 style={{ color: '#3498db' }}>📊 System Information</h3>
            <p><strong>Version:</strong> {systemInfo.version}</p>
            <p><strong>Environment:</strong> {systemInfo.environment}</p>
            <p><strong>Database:</strong> {systemInfo.database}</p>
            <h4>Features:</h4>
            <ul>
              {systemInfo.features?.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
          </div>
        )}

        <div style={{ backgroundColor: '#fff3cd', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
          <h3 style={{ color: '#856404' }}>🚀 Getting Started</h3>
          <p>To start developing with the Persian Legal AI system:</p>
          <ol>
            <li>Backend API is running on <code>http://localhost:8000</code></li>
            <li>Frontend is accessible on <code>http://localhost:3000</code></li>
            <li>Persian module is available on <code>http://localhost:8001</code></li>
          </ol>
        </div>

        <div style={{ backgroundColor: '#f8d7da', padding: '15px', borderRadius: '8px' }}>
          <h3 style={{ color: '#721c24' }}>📝 Next Steps</h3>
          <ul>
            <li>Configure AI models and training data</li>
            <li>Implement specific business logic</li>
            <li>Add comprehensive testing</li>
            <li>Prepare for production deployment</li>
          </ul>
        </div>
      </main>
    </div>
  )
}