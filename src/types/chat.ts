export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  type: 'text' | 'image' | 'document';
  fileData?: {
    name: string;
    type: string;
    size: number;
    url?: string;
  };
  isLoading?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface FileUpload {
  file: File;
  preview?: string;
  type: 'image' | 'document';
}