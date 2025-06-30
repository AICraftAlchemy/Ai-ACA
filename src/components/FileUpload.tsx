import React, { useCallback } from 'react';
import { Upload, X, Image, FileText, File } from 'lucide-react';
import { FileUpload as FileUploadType } from '../types/chat';
import { isImageFile, isPDFFile, isTextFile, formatFileSize } from '../utils/fileProcessing';

interface FileUploadProps {
  onFileSelect: (file: FileUploadType) => void;
  onFileRemove: () => void;
  selectedFile: FileUploadType | null;
  className?: string;
}

export default function FileUpload({ 
  onFileSelect, 
  onFileRemove, 
  selectedFile, 
  className = '' 
}: FileUploadProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFileSelection(files[0]);
      }
    },
    [onFileSelect]
  );

  const handleFileSelection = useCallback(
    (file: File) => {
      if (isImageFile(file)) {
        const reader = new FileReader();
        reader.onload = () => {
          onFileSelect({
            file,
            preview: reader.result as string,
            type: 'image'
          });
        };
        reader.readAsDataURL(file);
      } else if (isPDFFile(file) || isTextFile(file)) {
        onFileSelect({
          file,
          type: 'document'
        });
      } else {
        alert('Unsupported file type. Please upload images, PDFs, or text files.');
      }
    },
    [onFileSelect]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const getFileIcon = (file: FileUploadType) => {
    if (file.type === 'image') return <Image className="w-5 h-5 text-blue-400" />;
    if (isPDFFile(file.file)) return <File className="w-5 h-5 text-red-400" />;
    return <FileText className="w-5 h-5 text-green-400" />;
  };

  if (selectedFile) {
    return (
      <div className={`relative w-full max-w-full ${className}`}>
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 p-3 sm:p-4 bg-gray-800/80 rounded-xl border border-gray-700/50 backdrop-blur-sm w-full">
          <div className="flex-shrink-0">
            {getFileIcon(selectedFile)}
          </div>
          <div className="flex-1 min-w-0 w-full">
            <p className="text-sm font-medium text-gray-200 truncate">
              {selectedFile.file.name}
            </p>
            <p className="text-xs text-gray-400">
              {formatFileSize(selectedFile.file.size)} • {selectedFile.file.type}
            </p>
          </div>
          <button
            onClick={onFileRemove}
            className="flex-shrink-0 p-1.5 text-gray-400 hover:text-red-400 transition-colors rounded-lg hover:bg-red-500/10"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        {selectedFile.preview && (
          <div className="mt-2 sm:mt-3 w-full flex justify-center">
            <img
              src={selectedFile.preview}
              alt="Preview"
              className="max-w-full max-h-40 rounded-xl object-cover shadow-lg"
            />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`w-full max-w-full ${className}`}>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="border-2 border-dashed border-gray-600/50 rounded-xl p-3 sm:p-4 md:p-6 text-center hover:border-gray-500/50 transition-all duration-200 cursor-pointer bg-gray-800/30 backdrop-blur-sm w-full"
        onClick={() => document.getElementById('file-upload')?.click()}
      >
        <Upload className="w-8 h-8 text-gray-400 mx-auto mb-3" />
        <p className="text-sm text-gray-300 mb-2 font-medium">
          Drop files here or click to upload
        </p>
        <p className="text-xs text-gray-500">
          Supports images, PDFs, text files, and documents
        </p>
        <div className="flex items-center justify-center space-x-4 mt-4 text-xs text-gray-400">
          <div className="flex items-center space-x-1">
            <Image className="w-3 h-3" />
            <span>Images</span>
          </div>
          <div className="flex items-center space-x-1">
            <File className="w-3 h-3" />
            <span>PDFs</span>
          </div>
          <div className="flex items-center space-x-1">
            <FileText className="w-3 h-3" />
            <span>Documents</span>
          </div>
        </div>
        <input
          id="file-upload"
          type="file"
          accept="image/*,.pdf,.txt,.md,.doc,.docx"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) {
              handleFileSelection(file);
            }
          }}
          className="hidden"
        />
      </div>
    </div>
  );
}