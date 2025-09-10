// route-updater.js
const fs = require('fs');
const path = require('path');

class RouteUpdater {
  static updateDashboardRoutes() {
    console.log('🛣️  Updating dashboard routes...');
    
    const dashboardPath = '/workspace/frontend/src/components/CompletePersianAIDashboard.tsx';
    let dashboardContent = fs.readFileSync(dashboardPath, 'utf8');
    
    // Add Bolt imports
    const boltImports = `
// Bolt Components
import BoltAnalyticsPage from './bolt/pages/analytics-page';
import BoltDataPage from './bolt/pages/data-page';
import BoltModelsPage from './bolt/pages/models-page';
import BoltMonitoringPage from './bolt/pages/monitoring-page';
import BoltSettingsPage from './bolt/pages/settings-page';
import BoltLogsPage from './bolt/pages/logs-page';
import BoltTeam from './bolt/components/team';
import { BoltProvider } from '../services/boltContext';
import { boltApi } from '../api/boltApi';
`;

    // Find the last import line
    const importLines = dashboardContent.split('\n').filter(line => line.trim().startsWith('import'));
    const lastImportIndex = dashboardContent.lastIndexOf(importLines[importLines.length - 1]);
    const importEndPoint = dashboardContent.indexOf('\n', lastImportIndex);
    
    dashboardContent = 
      dashboardContent.slice(0, importEndPoint) + 
      boltImports + 
      dashboardContent.slice(importEndPoint);

    // Add Bolt routes to renderContent function
    const boltRoutes = `
      // Bolt Routes
      case 'bolt-analytics':
        return <BoltAnalyticsPage />;
      case 'bolt-data':
        return <BoltDataPage />;
      case 'bolt-models':
        return <BoltModelsPage />;
      case 'bolt-monitoring':
        return <BoltMonitoringPage />;
      case 'bolt-settings':
        return <BoltSettingsPage />;
      case 'bolt-logs':
        return <BoltLogsPage />;
      case 'bolt-team':
        return <BoltTeam />;
`;

    // Find the renderContent function and add routes before default case
    const renderContentStart = dashboardContent.indexOf('const renderContent = () => {');
    if (renderContentStart !== -1) {
      const switchStart = dashboardContent.indexOf('switch (activeTab)', renderContentStart);
      const defaultCaseIndex = dashboardContent.indexOf('default:', switchStart);
      
      if (defaultCaseIndex !== -1) {
        dashboardContent = 
          dashboardContent.slice(0, defaultCaseIndex) + 
          boltRoutes + 
          '      ' + // indent
          dashboardContent.slice(defaultCaseIndex);
      }
    }

    // Add Bolt navigation items
    const boltNavigation = `
                {/* Bolt Navigation */}
                <div className="relative group">
                  <button
                    onClick={() => setActiveTab('bolt-analytics')}
                    className={\`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium \${
                      activeTab.startsWith('bolt') 
                        ? 'border-purple-500 text-purple-600' 
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }\`}
                  >
                    ⚡ Bolt
                  </button>
                  
                  {/* Bolt Submenu */}
                  <div className="absolute top-full left-0 mt-1 bg-white shadow-lg rounded-md border border-gray-200 hidden group-hover:block z-10 min-w-48">
                    <button
                      onClick={() => setActiveTab('bolt-analytics')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      📊 تحلیلات
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-data')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      📄 داده‌ها
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-models')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      🤖 مدل‌ها
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-monitoring')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      📈 نظارت
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-logs')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      📋 لاگ‌ها
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-team')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      👥 تیم
                    </button>
                    <button
                      onClick={() => setActiveTab('bolt-settings')}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 w-full text-right"
                    >
                      ⚙️ تنظیمات
                    </button>
                  </div>
                </div>
`;

    // Find navigation section and add Bolt navigation
    const navSectionPattern = /نظارت سیستم.*?<\/button>/s;
    const navMatch = dashboardContent.match(navSectionPattern);
    
    if (navMatch) {
      const insertPoint = dashboardContent.indexOf(navMatch[0]) + navMatch[0].length;
      dashboardContent = 
        dashboardContent.slice(0, insertPoint) + 
        boltNavigation + 
        dashboardContent.slice(insertPoint);
    }

    // Wrap return with BoltProvider
    const returnPattern = /return\s*\(/;
    const returnMatch = dashboardContent.match(returnPattern);
    if (returnMatch) {
      const returnIndex = returnMatch.index;
      const openParenIndex = dashboardContent.indexOf('(', returnIndex);
      
      // Find the matching closing parenthesis
      let parenCount = 1;
      let closeParenIndex = openParenIndex + 1;
      while (parenCount > 0 && closeParenIndex < dashboardContent.length) {
        if (dashboardContent[closeParenIndex] === '(') parenCount++;
        if (dashboardContent[closeParenIndex] === ')') parenCount--;
        closeParenIndex++;
      }
      closeParenIndex--; // Back to the actual closing paren
      
      // Wrap content with BoltProvider
      const beforeReturn = dashboardContent.slice(0, openParenIndex + 1);
      const returnContent = dashboardContent.slice(openParenIndex + 1, closeParenIndex);
      const afterReturn = dashboardContent.slice(closeParenIndex);
      
      dashboardContent = beforeReturn + 
        '\n    <BoltProvider>\n' +
        returnContent + 
        '\n    </BoltProvider>\n' +
        afterReturn;
    }

    fs.writeFileSync(dashboardPath, dashboardContent, 'utf8');
    console.log('✅ Dashboard routes updated');
  }

  static createBoltErrorBoundary() {
    console.log('🛡️  Creating Bolt error boundary...');
    
    const errorBoundaryContent = `import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class BoltErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Bolt Error Boundary caught an error:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-100 rounded-full">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
            <div className="mt-4 text-center">
              <h3 className="text-lg font-medium text-gray-900">
                خطای Bolt
              </h3>
              <p className="mt-2 text-sm text-gray-500">
                متأسفانه خطایی در سیستم Bolt رخ داده است.
              </p>
              {this.state.error && (
                <details className="mt-4 text-xs text-left bg-gray-50 p-3 rounded">
                  <summary className="cursor-pointer font-medium">جزئیات خطا</summary>
                  <pre className="mt-2 whitespace-pre-wrap">{this.state.error.message}</pre>
                </details>
              )}
            </div>
            <div className="mt-6">
              <button
                onClick={this.handleRetry}
                className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                تلاش مجدد
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}`;

    fs.writeFileSync('/workspace/frontend/src/components/bolt/BoltErrorBoundary.tsx', errorBoundaryContent);
    console.log('✅ Bolt error boundary created');
  }

  static updateAppRouter() {
    console.log('🔀 Checking App.tsx for router updates...');
    
    const appPath = '/workspace/frontend/src/App.tsx';
    if (fs.existsSync(appPath)) {
      let appContent = fs.readFileSync(appPath, 'utf8');
      
      // Add Bolt error boundary if needed
      if (!appContent.includes('BoltErrorBoundary')) {
        const errorBoundaryImport = "import { BoltErrorBoundary } from './components/bolt/BoltErrorBoundary';\n";
        
        // Add import
        const lastImport = appContent.lastIndexOf('import');
        const importEnd = appContent.indexOf('\n', lastImport);
        appContent = appContent.slice(0, importEnd) + '\n' + errorBoundaryImport + appContent.slice(importEnd);
        
        // Find the main div and wrap with error boundary
        const appDivPattern = /<div className="App">/;
        const appDivMatch = appContent.match(appDivPattern);
        
        if (appDivMatch) {
          appContent = appContent.replace(
            '<div className="App">',
            '<BoltErrorBoundary>\n      <div className="App">'
          );
          
          // Find the closing div and add error boundary close
          const lastDivIndex = appContent.lastIndexOf('</div>');
          if (lastDivIndex !== -1) {
            appContent = appContent.slice(0, lastDivIndex) + 
              '</div>\n    </BoltErrorBoundary>\n' +
              appContent.slice(lastDivIndex + 6);
          }
        }
        
        fs.writeFileSync(appPath, appContent, 'utf8');
        console.log('✅ App.tsx updated with Bolt error boundary');
      }
    }
  }
}

// Main execution
async function main() {
  try {
    RouteUpdater.createBoltErrorBoundary();
    RouteUpdater.updateDashboardRoutes();
    RouteUpdater.updateAppRouter();
    
    console.log('🎉 Route integration completed successfully!');
    
  } catch (error) {
    console.error('❌ Error updating routes:', error);
    process.exit(1);
  }
}

main();