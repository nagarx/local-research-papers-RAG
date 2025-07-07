'use client';

import React, { useState } from 'react';
import { Settings, Server, Brain, Database, Sliders } from 'lucide-react';
import { cn } from '@/lib/utils';
import { SystemConfig } from '@/lib/api';

interface SettingsPanelProps {
  config?: SystemConfig;
  onConfigUpdate?: (updates: Partial<SystemConfig>) => void;
  onPreloadModels?: () => void;
  onInitializeSystem?: () => void;
}

export function SettingsPanel({
  config,
  onConfigUpdate,
  onPreloadModels,
  onInitializeSystem,
}: SettingsPanelProps) {
  const [activeSection, setActiveSection] = useState<string>('system');

  const sections = [
    { id: 'system', label: 'System', icon: Server },
    { id: 'llm', label: 'LLM Settings', icon: Brain },
    { id: 'chunking', label: 'Chunking', icon: Database },
    { id: 'retrieval', label: 'Retrieval', icon: Sliders },
  ];

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-lg font-semibold mb-4">Settings</h2>
      
      {/* Section Tabs */}
      <div className="space-y-1 mb-6">
        {sections.map((section) => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors",
              activeSection === section.id
                ? "bg-primary text-primary-foreground"
                : "hover:bg-muted"
            )}
          >
            <section.icon className="w-4 h-4" />
            {section.label}
          </button>
        ))}
      </div>

      {/* Settings Content */}
      <div className="flex-1 space-y-4">
        {activeSection === 'system' && (
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-2">System Control</h3>
              <div className="space-y-2">
                <button
                  onClick={onInitializeSystem}
                  className="w-full px-3 py-2 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  Initialize System
                </button>
                <button
                  onClick={onPreloadModels}
                  className="w-full px-3 py-2 text-sm border border-border rounded-md hover:bg-muted"
                >
                  Pre-load Models
                </button>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'llm' && config && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Model</label>
              <select
                value={config.llm.model}
                onChange={(e) => onConfigUpdate?.({ llm: { ...config.llm, model: e.target.value } })}
                className="w-full mt-1 px-3 py-2 text-sm border border-border rounded-md bg-background"
              >
                <option value="llama3.2:latest">Llama 3.2</option>
                <option value="qwen2.5-coder:latest">Qwen 2.5 Coder</option>
                <option value="deepseek-r1:7b">DeepSeek R1</option>
              </select>
            </div>
            
            <div>
              <label className="text-sm font-medium">Temperature</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.llm.temperature}
                onChange={(e) => onConfigUpdate?.({ llm: { ...config.llm, temperature: parseFloat(e.target.value) } })}
                className="w-full mt-1"
              />
              <span className="text-xs text-muted-foreground">{config.llm.temperature}</span>
            </div>

            <div>
              <label className="text-sm font-medium">Context Length</label>
              <input
                type="number"
                value={config.llm.context_length}
                onChange={(e) => onConfigUpdate?.({ llm: { ...config.llm, context_length: parseInt(e.target.value) } })}
                className="w-full mt-1 px-3 py-2 text-sm border border-border rounded-md bg-background"
              />
            </div>
          </div>
        )}

        {activeSection === 'chunking' && config && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Chunk Size</label>
              <input
                type="range"
                min="500"
                max="2000"
                step="100"
                value={config.chunking.chunk_size}
                onChange={(e) => onConfigUpdate?.({ chunking: { ...config.chunking, chunk_size: parseInt(e.target.value) } })}
                className="w-full mt-1"
              />
              <span className="text-xs text-muted-foreground">{config.chunking.chunk_size} tokens</span>
            </div>

            <div>
              <label className="text-sm font-medium">Chunk Overlap</label>
              <input
                type="range"
                min="50"
                max="200"
                step="10"
                value={config.chunking.chunk_overlap}
                onChange={(e) => onConfigUpdate?.({ chunking: { ...config.chunking, chunk_overlap: parseInt(e.target.value) } })}
                className="w-full mt-1"
              />
              <span className="text-xs text-muted-foreground">{config.chunking.chunk_overlap} tokens</span>
            </div>
          </div>
        )}

        {activeSection === 'retrieval' && config && (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Top K Results</label>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={config.retrieval.top_k}
                onChange={(e) => onConfigUpdate?.({ retrieval: { ...config.retrieval, top_k: parseInt(e.target.value) } })}
                className="w-full mt-1"
              />
              <span className="text-xs text-muted-foreground">{config.retrieval.top_k}</span>
            </div>

            <div>
              <label className="text-sm font-medium">Similarity Threshold</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={config.retrieval.similarity_threshold}
                onChange={(e) => onConfigUpdate?.({ retrieval: { ...config.retrieval, similarity_threshold: parseFloat(e.target.value) } })}
                className="w-full mt-1"
              />
              <span className="text-xs text-muted-foreground">{config.retrieval.similarity_threshold}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}