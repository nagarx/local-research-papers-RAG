'use client';

import React, { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from '@/components/Layout';
import { SettingsPanel } from '@/components/SettingsPanel';
import { DocumentPanel } from '@/components/DocumentPanel';
import { MainPanel } from '@/components/MainPanel';
import { ragApi, DocumentInfo, ChatMessage, SystemConfig, AnalyticsOverview, SessionInfo } from '@/lib/api';

// Create a client
const queryClient = new QueryClient();

function RAGApp() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [config, setConfig] = useState<SystemConfig | undefined>();
  const [analytics, setAnalytics] = useState<AnalyticsOverview | undefined>();
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | undefined>();
  const [loading, setLoading] = useState(true);
  const [systemInitialized, setSystemInitialized] = useState(false);

  // Initialize system on mount
  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      setLoading(true);
      
      // Check system health
      const healthResponse = await ragApi.getHealth();
      if (healthResponse.data.overall_status === 'healthy') {
        setSystemInitialized(true);
        
        // Load initial data
        await Promise.all([
          loadDocuments(),
          loadConfig(),
          loadAnalytics(),
          checkSession(),
        ]);
      }
    } catch (error) {
      console.error('Failed to initialize app:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async () => {
    try {
      const response = await ragApi.getAllDocuments();
      setDocuments(response.data);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const loadConfig = async () => {
    try {
      const response = await ragApi.getConfig();
      setConfig(response.data);
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await ragApi.getAnalytics();
      setAnalytics(response.data);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const checkSession = async () => {
    try {
      const response = await ragApi.getCurrentSession();
      if (response.data) {
        setSessionInfo(response.data);
      }
    } catch (error) {
      console.error('Failed to check session:', error);
    }
  };

  const handleInitializeSystem = async () => {
    try {
      await initializeApp();
    } catch (error) {
      console.error('Failed to initialize system:', error);
    }
  };

  const handlePreloadModels = async () => {
    try {
      await ragApi.preloadModels();
      alert('Models preloaded successfully!');
    } catch (error) {
      console.error('Failed to preload models:', error);
      alert('Failed to preload models');
    }
  };

  const handleConfigUpdate = async (updates: Partial<SystemConfig>) => {
    try {
      await ragApi.updateConfig(updates);
      await loadConfig();
    } catch (error) {
      console.error('Failed to update config:', error);
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    try {
      await ragApi.deleteDocument(documentId);
      await loadDocuments();
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const handleFileUpload = async (files: File[], storageType: 'permanent' | 'temporary') => {
    try {
      // Start a session if not already started
      if (!sessionInfo) {
        const sessionResponse = await ragApi.startSession();
        setSessionInfo(sessionResponse.data);
      }

      await ragApi.uploadFiles(files, storageType);
      await loadDocuments();
      await loadAnalytics();
      alert('Files uploaded successfully!');
    } catch (error) {
      console.error('Failed to upload files:', error);
      alert('Failed to upload files');
    }
  };

  const handleSendMessage = async (message: string) => {
    try {
      // Add user message immediately
      const userMessage: ChatMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Send to API
      const response = await ragApi.sendChatMessage(
        message,
        sessionInfo?.session_id
      );

      // Add assistant response
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.data.response,
        sources: response.data.sources,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Add error message
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  if (loading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading RAG System...</p>
        </div>
      </div>
    );
  }

  return (
    <Layout
      leftPanel={
        <SettingsPanel
          config={config}
          onConfigUpdate={handleConfigUpdate}
          onPreloadModels={handlePreloadModels}
          onInitializeSystem={handleInitializeSystem}
        />
      }
      centerPanel={
        <MainPanel
          messages={messages}
          analytics={analytics}
          onSendMessage={handleSendMessage}
          onFileUpload={handleFileUpload}
          sessionInfo={sessionInfo}
        />
      }
      rightPanel={
        <DocumentPanel
          documents={documents}
          onDeleteDocument={handleDeleteDocument}
          onRefresh={loadDocuments}
        />
      }
    />
  );
}

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <RAGApp />
    </QueryClientProvider>
  );
}
