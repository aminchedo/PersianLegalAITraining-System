import React, { useState, useEffect } from 'react';
import { TrainingSessionRequest, TrainingSessionStatus } from '../../types/training';
import trainingService from '../../services/trainingService';
import authService from '../../services/authService';

const TrainingControlPanel: React.FC = () => {
  const [sessions, setSessions] = useState<TrainingSessionStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [creating, setCreating] = useState(false);

  const [newSession, setNewSession] = useState<TrainingSessionRequest>({
    model_type: 'dora',
    model_name: '',
    config: {
      learning_rate: 0.001,
      batch_size: 8,
      epochs: 3,
      max_length: 512
    },
    data_source: 'sample',
    task_type: 'text_classification'
  });

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const sessionsData = await trainingService.getTrainingSessions();
      setSessions(sessionsData);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);

    try {
      await trainingService.createTrainingSession(newSession);
      setShowCreateForm(false);
      setNewSession({
        model_type: 'dora',
        model_name: '',
        config: {
          learning_rate: 0.001,
          batch_size: 8,
          epochs: 3,
          max_length: 512
        },
        data_source: 'sample',
        task_type: 'text_classification'
      });
      await fetchSessions();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleStopSession = async (sessionId: string) => {
    try {
      await trainingService.stopTrainingSession(sessionId);
      await fetchSessions();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (window.confirm('Are you sure you want to delete this training session?')) {
      try {
        await trainingService.deleteTrainingSession(sessionId);
        await fetchSessions();
      } catch (err: any) {
        setError(err.message);
      }
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'stopped':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const canCreateTraining = authService.hasPermission('training');
  const canManageTraining = authService.hasPermission('admin') || authService.hasPermission('training');

  if (loading) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="space-y-3">
              <div className="h-3 bg-gray-200 rounded"></div>
              <div className="h-3 bg-gray-200 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Training Sessions</h3>
          {canCreateTraining && (
            <button
              onClick={() => setShowCreateForm(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Create Training Session
            </button>
          )}
        </div>

        {error && (
          <div className="mb-4 rounded-md bg-red-50 p-4">
            <div className="text-sm text-red-700">{error}</div>
          </div>
        )}

        {showCreateForm && (
          <div className="mb-6 p-4 border border-gray-200 rounded-lg">
            <h4 className="text-md font-medium text-gray-900 mb-4">Create New Training Session</h4>
            <form onSubmit={handleCreateSession} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Model Name</label>
                  <input
                    type="text"
                    value={newSession.model_name}
                    onChange={(e) => setNewSession(prev => ({ ...prev, model_name: e.target.value }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Model Type</label>
                  <select
                    value={newSession.model_type}
                    onChange={(e) => setNewSession(prev => ({ ...prev, model_type: e.target.value as 'dora' | 'qr_adaptor' }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  >
                    <option value="dora">DoRA</option>
                    <option value="qr_adaptor">QR Adaptor</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Data Source</label>
                  <select
                    value={newSession.data_source}
                    onChange={(e) => setNewSession(prev => ({ ...prev, data_source: e.target.value as 'sample' | 'qavanin' | 'majlis' }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  >
                    <option value="sample">Sample Data</option>
                    <option value="qavanin">Qavanin</option>
                    <option value="majlis">Majlis</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Task Type</label>
                  <select
                    value={newSession.task_type}
                    onChange={(e) => setNewSession(prev => ({ ...prev, task_type: e.target.value as 'text_classification' | 'question_answering' | 'text_generation' }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  >
                    <option value="text_classification">Text Classification</option>
                    <option value="question_answering">Question Answering</option>
                    <option value="text_generation">Text Generation</option>
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={newSession.config.learning_rate}
                    onChange={(e) => setNewSession(prev => ({ 
                      ...prev, 
                      config: { ...prev.config, learning_rate: parseFloat(e.target.value) }
                    }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Batch Size</label>
                  <input
                    type="number"
                    value={newSession.config.batch_size}
                    onChange={(e) => setNewSession(prev => ({ 
                      ...prev, 
                      config: { ...prev.config, batch_size: parseInt(e.target.value) }
                    }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Epochs</label>
                  <input
                    type="number"
                    value={newSession.config.epochs}
                    onChange={(e) => setNewSession(prev => ({ 
                      ...prev, 
                      config: { ...prev.config, epochs: parseInt(e.target.value) }
                    }))}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50"
                >
                  {creating ? 'Creating...' : 'Create Session'}
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="space-y-4">
          {sessions.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No training sessions found. Create one to get started.
            </div>
          ) : (
            sessions.map((session) => (
              <div key={session.session_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <h4 className="text-sm font-medium text-gray-900">{session.session_id}</h4>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(session.status)}`}>
                      {session.status}
                    </span>
                  </div>
                  {canManageTraining && (
                    <div className="flex space-x-2">
                      {session.status === 'running' && (
                        <button
                          onClick={() => handleStopSession(session.session_id)}
                          className="text-sm text-red-600 hover:text-red-800"
                        >
                          Stop
                        </button>
                      )}
                      {session.status !== 'running' && (
                        <button
                          onClick={() => handleDeleteSession(session.session_id)}
                          className="text-sm text-red-600 hover:text-red-800"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
                  <div>
                    <span className="font-medium">Progress:</span> {session.progress.current_epoch}/{session.progress.total_epochs} epochs
                  </div>
                  <div>
                    <span className="font-medium">Samples:</span> {session.progress.train_samples} train, {session.progress.eval_samples} eval
                  </div>
                  <div>
                    <span className="font-medium">Created:</span> {new Date(session.created_at).toLocaleString()}
                  </div>
                </div>

                {session.status === 'running' && (
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${session.progress.total_epochs > 0 ? (session.progress.current_epoch / session.progress.total_epochs) * 100 : 0}%` 
                        }}
                      ></div>
                    </div>
                  </div>
                )}

                {session.error_message && (
                  <div className="mt-2 text-sm text-red-600">
                    Error: {session.error_message}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingControlPanel;