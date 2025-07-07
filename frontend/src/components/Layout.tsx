'use client';

import React from 'react';
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from 'react-resizable-panels';
import { cn } from '@/lib/utils';

interface LayoutProps {
  leftPanel: React.ReactNode;
  centerPanel: React.ReactNode;
  rightPanel: React.ReactNode;
}

export function Layout({ leftPanel, centerPanel, rightPanel }: LayoutProps) {
  return (
    <div className="h-screen w-screen overflow-hidden bg-background">
      <PanelGroup direction="horizontal" className="h-full">
        {/* Left Panel - Settings */}
        <Panel 
          defaultSize={20} 
          minSize={15} 
          maxSize={30}
          className={cn(
            "bg-muted/30 border-r border-border",
            "animate-in slide-in duration-300"
          )}
        >
          <div className="h-full overflow-y-auto p-4">
            {leftPanel}
          </div>
        </Panel>

        <PanelResizeHandle className="w-px bg-border hover:bg-primary/20 transition-colors" />

        {/* Center Panel - Main Content */}
        <Panel defaultSize={60} minSize={40}>
          <div className="h-full overflow-y-auto">
            {centerPanel}
          </div>
        </Panel>

        <PanelResizeHandle className="w-px bg-border hover:bg-primary/20 transition-colors" />

        {/* Right Panel - Document Management */}
        <Panel 
          defaultSize={20} 
          minSize={15} 
          maxSize={30}
          className={cn(
            "bg-muted/30 border-l border-border",
            "animate-in slide-in duration-300"
          )}
        >
          <div className="h-full overflow-y-auto p-4">
            {rightPanel}
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}