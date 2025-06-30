import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, X, Sparkles } from 'lucide-react';
import FileUpload from './FileUpload';
import { FileUpload as FileUploadType } from '../types/chat';

interface ChatInputProps {
  onSendMessage: (message: string, file?: FileUploadType) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ 
  onSendMessage, 
  disabled = false,
  placeholder = "Ask AI-ACA anything..." 
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<FileUploadType | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((message.trim() || selectedFile) && !disabled) {
      onSendMessage(message.trim(), selectedFile || undefined);
      setMessage('');
      setSelectedFile(null);
      setShowFileUpload(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  return (
    <div className="border-t border-gray-800/50 bg-gray-900/80 backdrop-blur-sm p-2 sm:p-3 md:p-4 sticky bottom-0 z-30">
      {showFileUpload && (
        <div className="mb-4">
          <FileUpload
            onFileSelect={setSelectedFile}
            onFileRemove={() => setSelectedFile(null)}
            selectedFile={selectedFile}
          />
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row items-end gap-2 sm:space-x-3">
        <div className="flex-1 relative w-full">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="w-full bg-gray-800/80 border border-gray-700/50 rounded-xl px-4 py-3 pr-12 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 max-h-32 overflow-y-auto backdrop-blur-sm transition-all duration-200"
          />
          <button
            type="button"
            onClick={() => setShowFileUpload(!showFileUpload)}
            className={`absolute right-3 top-3 p-1.5 rounded-lg transition-all duration-200 ${
              showFileUpload || selectedFile
                ? 'text-blue-400 bg-blue-400/20 border border-blue-500/30'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-700/30'
            }`}
          >
            {showFileUpload ? <X className="w-4 h-4" /> : <Paperclip className="w-4 h-4" />}
          </button>
        </div>

        <button
          type="submit"
          disabled={disabled || (!message.trim() && !selectedFile)}
          className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed text-white rounded-xl transition-all duration-200 flex items-center justify-center shadow-lg"
        >
          {disabled ? (
            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </form>

      {selectedFile && (
        <div className="mt-3 flex items-center space-x-2 text-xs text-gray-400">
          <Sparkles className="w-3 h-3 text-blue-400" />
          <span>File ready: <span className="text-blue-400">{selectedFile.file.name}</span></span>
        </div>
      )}
    </div>
  );
}