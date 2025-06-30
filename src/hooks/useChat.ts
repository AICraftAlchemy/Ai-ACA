import { useState, useCallback } from 'react';
import { Message, ChatSession, FileUpload } from '../types/chat';
import { generateResponse, fileToGenerativePart } from '../utils/gemini';
import { processPDFFile, processTextFile, isImageFile, isPDFFile, isTextFile } from '../utils/fileProcessing';

export function useChat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getCurrentSession = useCallback(() => {
    return sessions.find(session => session.id === currentSessionId);
  }, [sessions, currentSessionId]);

  const createNewSession = useCallback(() => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setError(null);
    
    return newSession.id;
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => {
      const newSessions = prev.filter(session => session.id !== sessionId);
      
      // If we're deleting the current session, switch to another one
      if (currentSessionId === sessionId) {
        if (newSessions.length > 0) {
          setCurrentSessionId(newSessions[0].id);
        } else {
          setCurrentSessionId(null);
        }
      }
      
      return newSessions;
    });
  }, [currentSessionId]);

  const clearAllSessions = useCallback(() => {
    setSessions([]);
    setCurrentSessionId(null);
    setError(null);
  }, []);

  const updateSessionTitle = useCallback((sessionId: string, title: string) => {
    setSessions(prev => prev.map(session => 
      session.id === sessionId 
        ? { ...session, title, updatedAt: new Date() }
        : session
    ));
  }, []);

  const sendMessage = useCallback(async (content: string, file?: FileUpload) => {
    if (!currentSessionId) {
      const newSessionId = createNewSession();
      // Wait for the session to be created
      setTimeout(() => {
        sendMessage(content, file);
      }, 100);
      return;
    }

    setIsLoading(true);
    setError(null);

    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date(),
      type: file ? (file.type === 'image' ? 'image' : 'document') : 'text',
      fileData: file ? {
        name: file.file.name,
        type: file.file.type,
        size: file.file.size,
        url: file.preview
      } : undefined
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      type: 'text',
      isLoading: true
    };

    // Add user message and loading message
    setSessions(prev => prev.map(session => 
      session.id === currentSessionId
        ? { 
            ...session, 
            messages: [...session.messages, userMessage, loadingMessage],
            updatedAt: new Date()
          }
        : session
    ));

    try {
      let fileData;
      let enhancedPrompt = content;

      // Detect if the prompt is a mathematical problem
      const isMathProblem = /\b(solve|calculate|integrate|differentiate|find|equation|math|\d+\s*[+\-*/^=])\b/i.test(content);

      if (file) {
        if (isImageFile(file.file)) {
          fileData = await fileToGenerativePart(file.file);
          enhancedPrompt = content || "Please analyze this image in detail and describe what you see, including any text, objects, people, scenes, colors, and other relevant details.";
        } else if (isPDFFile(file.file)) {
          const pdfText = await processPDFFile(file.file);
          enhancedPrompt = `You are an expert document analyzer. I have uploaded a PDF document. Here is the extracted content from the document:
\n---DOCUMENT CONTENT START---\n${pdfText}\n---DOCUMENT CONTENT END---\n\nIf the task is to extract the context in the page, extract exactly what is in the page, preserving the original format, not in one line. If the user asks a question, analyze the file carefully and provide a 100% accurate, detailed, and well-formatted response.\n\nUser request: ${content || "Please extract and present the content of this page exactly as it appears, preserving formatting, and provide any additional analysis if needed."}`;
        } else if (isTextFile(file.file)) {
          const textContent = await processTextFile(file.file);
          enhancedPrompt = `You are an expert document analyzer. I have uploaded a text document. Here is the content:\n\n---DOCUMENT CONTENT START---\n${textContent}\n---DOCUMENT CONTENT END---\n\nIf the task is to extract the context in the page, extract exactly what is in the page, preserving the original format, not in one line. If the user asks a question, analyze the file carefully and provide a 100% accurate, detailed, and well-formatted response.\n\nUser request: ${content || "Please extract and present the content of this page exactly as it appears, preserving formatting, and provide any additional analysis if needed."}`;
        }
      } else if (isMathProblem) {
        enhancedPrompt = `You are a mathematics expert. Solve the following problem step by step, showing all intermediate steps and reasoning, and provide the most accurate answer possible.\n\nProblem: ${content}`;
      }

      // Get conversation history for context (limit to last 10 messages for better performance)
      const currentSession = sessions.find(s => s.id === currentSessionId);
      const conversationHistory = currentSession?.messages
        .filter(m => !m.isLoading)
        .slice(-10) // Keep only last 10 messages for context
        .map(m => ({
          role: m.role === 'user' ? 'user' : 'model',
          parts: [{ text: m.content }]
        })) || [];

      const response = await generateResponse(enhancedPrompt, fileData, conversationHistory);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        role: 'assistant',
        timestamp: new Date(),
        type: 'text'
      };

      // Update session title if it's the first message
      const updatedSession = sessions.find(s => s.id === currentSessionId);
      if (updatedSession && updatedSession.messages.filter(m => !m.isLoading).length <= 1) {
        const title = content.length > 40 ? content.substring(0, 40) + '...' : content;
        updateSessionTitle(currentSessionId, title || 'New Conversation');
      }

      // Replace loading message with actual response
      setSessions(prev => prev.map(session => 
        session.id === currentSessionId
          ? { 
              ...session, 
              messages: [...session.messages.filter(m => m.id !== loadingMessage.id), assistantMessage],
              updatedAt: new Date()
            }
          : session
      ));

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred while processing your request.';
      setError(errorMessage);
      
      // Remove loading message on error
      setSessions(prev => prev.map(session => 
        session.id === currentSessionId
          ? { 
              ...session, 
              messages: session.messages.filter(m => m.id !== loadingMessage.id),
              updatedAt: new Date()
            }
          : session
      ));
    } finally {
      setIsLoading(false);
    }
  }, [currentSessionId, sessions, createNewSession, updateSessionTitle]);

  return {
    sessions,
    currentSession: getCurrentSession(),
    currentSessionId,
    isLoading,
    error,
    createNewSession,
    deleteSession,
    clearAllSessions,
    sendMessage,
    setCurrentSessionId
  };
}