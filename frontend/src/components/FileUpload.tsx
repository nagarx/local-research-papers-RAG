'use client';

import React, { useState, useCallback } from 'react';
import { Upload, FileText, X, HardDrive, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  onUpload: (files: File[], storageType: 'permanent' | 'temporary') => void;
}

export function FileUpload({ onUpload }: FileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [storageType, setStorageType] = useState<'permanent' | 'temporary'>('temporary');
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      file => file.type === 'application/pdf'
    );
    setFiles(prev => [...prev, ...droppedFiles]);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...selectedFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (files.length > 0) {
      onUpload(files, storageType);
      setFiles([]);
    }
  };

  return (
    <div className="h-full p-6">
      <div className="max-w-2xl mx-auto">
        {/* Storage Type Selection */}
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-3">Storage Type</h3>
          <div className="flex gap-3">
            <button
              onClick={() => setStorageType('temporary')}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-md border transition-colors",
                storageType === 'temporary'
                  ? "border-primary bg-primary/5 text-primary"
                  : "border-border hover:bg-muted"
              )}
            >
              <Clock className="w-4 h-4" />
              <span className="text-sm font-medium">Temporary</span>
            </button>
            <button
              onClick={() => setStorageType('permanent')}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-md border transition-colors",
                storageType === 'permanent'
                  ? "border-primary bg-primary/5 text-primary"
                  : "border-border hover:bg-muted"
              )}
            >
              <HardDrive className="w-4 h-4" />
              <span className="text-sm font-medium">Permanent</span>
            </button>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {storageType === 'temporary' 
              ? 'Files will be removed when the session ends'
              : 'Files will be stored permanently in the system'}
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50"
          )}
        >
          <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-sm font-medium mb-2">
            Drop PDF files here or click to browse
          </p>
          <p className="text-xs text-muted-foreground mb-4">
            Only PDF files are supported
          </p>
          <input
            type="file"
            multiple
            accept=".pdf"
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 cursor-pointer"
          >
            Select Files
          </label>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium mb-3">Selected Files ({files.length})</h3>
            <div className="space-y-2">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 border border-border rounded-md"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">{file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="p-1.5 hover:bg-destructive/10 rounded transition-colors"
                  >
                    <X className="w-4 h-4 text-destructive" />
                  </button>
                </div>
              ))}
            </div>

            {/* Upload Button */}
            <button
              onClick={handleUpload}
              className="w-full mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 font-medium"
            >
              Upload {files.length} {files.length === 1 ? 'File' : 'Files'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}