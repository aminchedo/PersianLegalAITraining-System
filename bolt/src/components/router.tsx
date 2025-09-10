import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './layout/Layout';
import CompletePersianAIDashboard from './CompletePersianAIDashboard';
import TrainingControlPanel from './TrainingControlPanel';
import AnalyticsPage from './analytics-page';
import DataPage from './data-page';
import ModelsPage from './models-page';
import MonitoringPage from './monitoring-page';
import LogsPage from './logs-page';
import TeamPage from './team';
import SettingsPage from './settings-page';

const Router: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<TrainingControlPanel />} />
        <Route path="dashboard" element={<CompletePersianAIDashboard />} />
        <Route path="training" element={<TrainingControlPanel />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="data" element={<DataPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="monitoring" element={<MonitoringPage />} />
        <Route path="logs" element={<LogsPage />} />
        <Route path="team" element={<TeamPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
};

export default Router;