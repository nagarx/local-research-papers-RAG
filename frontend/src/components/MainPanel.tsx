'use client';

import React, { useState } from 'react';
import { Upload, Send, BarChart3, MessageSquare, FileUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ChatMessage, Source, AnalyticsOverview } from '@/lib/api';
import { FileUpload } from './FileUpload';
import { ChatInterface } from './ChatInterface';
import { Analytics } from './Analytics';

interface MainPanelProps {
  messages: ChatMessage[];
  analytics?: AnalyticsOverview;
  onSendMessage: (message: string) => void;
  onFileUpload: (files: File[], storageType: 'permanent' | 'temporary') => void;
  sessionInfo?: { session_id: string };
}

export function MainPanel({
  messages,
  analytics,
  onSendMessage,
  onFileUpload,
  sessionInfo,
}: MainPanelProps) {
  const [activeTab, setActiveTab] = useState<'upload' | 'chat' | 'analytics'>('chat');

  const tabs = [
    { id: 'upload', label: 'Upload', icon: FileUp },
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-border p-4">
        <h1 className="text-2xl font-semibold mb-1">ArXiv RAG System</h1>
        <p className="text-sm text-muted-foreground">
          Chat with your research papers using local LLMs
        </p>
        {sessionInfo && (
          <div className="mt-2 text-xs text-muted-foreground">
            Session: {sessionInfo.session_id.slice(0, 8)}...
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={cn(
                "flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors relative",
                activeTab === tab.id
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {activeTab === tab.id && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'upload' && (
          <FileUpload onUpload={onFileUpload} />
        )}
        {activeTab === 'chat' && (
          <ChatInterface
            messages={messages}
            onSendMessage={onSendMessage}
          />
        )}
        {activeTab === 'analytics' && (
          <Analytics data={analytics} />
        )}
      </div>
    </div>
  );
}