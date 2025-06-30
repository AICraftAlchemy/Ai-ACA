import React, { useEffect, useRef } from 'react';
import { Sparkles, AlertCircle, Brain } from 'lucide-react';
import MessageBubble from './MessageBubble';
import { Message } from '../types/chat';

interface ChatAreaProps {
  messages: Message[];
  isLoading?: boolean;
  error?: string | null;
}

export default function ChatArea({ messages, isLoading, error }: ChatAreaProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  if (messages.length === 0 && !isLoading && !error) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-2xl">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-2xl">
            <Brain className="w-10 h-10 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-3">Welcome to AI-ACA</h2>
          <p className="text-gray-400 mb-8 text-lg leading-relaxed">
            Your advanced AI assistant with unlimited capabilities. Experience intelligent conversations, 
            comprehensive document analysis, and precise image understanding.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-gradient-to-br from-blue-600/10 to-purple-600/10 border border-blue-500/20 p-4 rounded-xl backdrop-blur-sm">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <p className="text-gray-300 font-medium mb-1">Unlimited Conversations</p>
              <p className="text-gray-400 text-xs">Ask anything without token limits</p>
            </div>
            <div className="bg-gradient-to-br from-green-600/10 to-emerald-600/10 border border-green-500/20 p-4 rounded-xl backdrop-blur-sm">
              <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-300 font-medium mb-1">Image Analysis</p>
              <p className="text-gray-400 text-xs">Upload and analyze any image</p>
            </div>
            <div className="bg-gradient-to-br from-purple-600/10 to-pink-600/10 border border-purple-500/20 p-4 rounded-xl backdrop-blur-sm">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-300 font-medium mb-1">Document Processing</p>
              <p className="text-gray-400 text-xs">PDFs, text files, and more</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-2 sm:p-4 md:p-6">
      <div className="max-w-full md:max-w-4xl mx-auto space-y-4 md:space-y-6">
        {error && (
          <div className="bg-gradient-to-r from-red-900/20 to-red-800/20 border border-red-500/30 rounded-xl p-4 flex items-center space-x-3 backdrop-blur-sm">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <div>
              <p className="text-red-300 font-medium">Error</p>
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}