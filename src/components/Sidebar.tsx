import React from 'react';
import { Plus, MessageSquare, Settings, Sparkles, Trash2, RotateCcw } from 'lucide-react';
import { ChatSession } from '../types/chat';

interface SidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onClearAll: () => void;
  isCollapsed: boolean;
}

export default function Sidebar({ 
  sessions, 
  currentSessionId, 
  onNewChat, 
  onSelectSession, 
  onDeleteSession,
  onClearAll,
  isCollapsed 
}: SidebarProps) {
  if (isCollapsed) {
    return (
      <div className="w-16 bg-gray-900/90 backdrop-blur-sm border-r border-gray-800/50 flex flex-col items-center py-4">
        <button
          onClick={onNewChat}
          className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 rounded-xl flex items-center justify-center transition-all duration-200 mb-4 shadow-lg"
        >
          <Plus className="w-5 h-5 text-white" />
        </button>
        <div className="flex flex-col space-y-2">
          {sessions.slice(0, 5).map((session) => (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 ${
                currentSessionId === session.id
                  ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-blue-400 border border-blue-500/30'
                  : 'text-gray-400 hover:bg-gray-800/50 hover:text-white'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-gray-900/90 backdrop-blur-sm border-r border-gray-800/50 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-800/50">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">AI-ACA</h1>
            <p className="text-xs text-gray-400">Advanced AI Assistant</p>
          </div>
        </div>
        <button
          onClick={onNewChat}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-4 py-3 rounded-xl flex items-center justify-center space-x-2 transition-all duration-200 shadow-lg"
        >
          <Plus className="w-4 h-4" />
          <span className="font-medium">New Conversation</span>
        </button>
      </div>

      {/* Chat Sessions */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group relative rounded-xl transition-all duration-200 ${
                currentSessionId === session.id
                  ? 'bg-gradient-to-r from-blue-600/10 to-purple-600/10 border border-blue-500/20 shadow-md'
                  : 'hover:bg-gray-800/30'
              }`}
            >
              <button
                onClick={() => onSelectSession(session.id)}
                className="w-full text-left p-3 rounded-xl"
              >
                <div className="flex items-start space-x-3">
                  <MessageSquare className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                    currentSessionId === session.id ? 'text-blue-400' : 'text-gray-400'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">
                      {session.title}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      {session.messages.length} messages • {session.updatedAt.toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(session.id);
                }}
                className="absolute top-2 right-2 p-1.5 text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all duration-200 rounded-lg hover:bg-red-500/10"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800/50 space-y-2">
        {sessions.length > 0 && (
          <button 
            onClick={onClearAll}
            className="w-full flex items-center space-x-3 p-3 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-all duration-200"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="text-sm font-medium">Clear All Chats</span>
          </button>
        )}
        <button className="w-full flex items-center space-x-3 p-3 text-gray-400 hover:text-white hover:bg-gray-800/30 rounded-xl transition-all duration-200">
          <Settings className="w-4 h-4" />
          <span className="text-sm font-medium">Settings</span>
        </button>
      </div>
    </div>
  );
}