import React, { useState } from 'react';
import { Menu, X, Sparkles } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';
import { useChat } from './hooks/useChat';

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const {
    sessions,
    currentSession,
    currentSessionId,
    isLoading,
    error,
    createNewSession,
    deleteSession,
    clearAllSessions,
    sendMessage,
    setCurrentSessionId
  } = useChat();

  // Create initial session if none exists
  React.useEffect(() => {
    if (sessions.length === 0) {
      createNewSession();
    }
  }, [sessions.length, createNewSession]);

  return (
    <div className="h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex flex-col md:flex-row overflow-hidden">
      {/* Mobile sidebar toggle */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className="fixed top-4 left-4 z-50 md:hidden p-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl shadow-lg backdrop-blur-sm"
      >
        {sidebarCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
      </button>

      {/* Sidebar */}
      <div className={`${sidebarCollapsed ? 'hidden md:block' : 'block'} w-full md:w-auto transition-all duration-300`}>
        <Sidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          onNewChat={createNewSession}
          onSelectSession={setCurrentSessionId}
          onDeleteSession={deleteSession}
          onClearAll={clearAllSessions}
          isCollapsed={sidebarCollapsed}
        />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 w-full">
        {/* Header */}
        <div className="border-b border-gray-800/50 p-3 md:p-4 bg-gray-900/80 backdrop-blur-sm">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="hidden md:block p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800/50 transition-all duration-200"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">
                  {currentSession?.title || 'AI-ACA'}
                </h1>
                <p className="text-sm text-gray-400">
                  Advanced AI Assistant
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Messages */}
        <ChatArea
          messages={currentSession?.messages || []}
          isLoading={isLoading}
          error={error}
        />

        {/* Chat Input */}
        <ChatInput
          onSendMessage={sendMessage}
          disabled={isLoading}
          placeholder="Ask AI-ACA anything... Upload images, documents, or PDFs for analysis!"
        />
      </div>

      {/* Mobile overlay */}
      {!sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden backdrop-blur-sm"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}
    </div>
  );
}

export default App;