'use client';

import React, { useState } from 'react';
import { FileText, Trash2, HardDrive, Clock, Search, Filter } from 'lucide-react';
import { cn } from '@/lib/utils';
import { DocumentInfo } from '@/lib/api';

interface DocumentPanelProps {
  documents: DocumentInfo[];
  onDeleteDocument: (documentId: string) => void;
  onRefresh: () => void;
}

export function DocumentPanel({ documents, onDeleteDocument, onRefresh }: DocumentPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'permanent' | 'temporary'>('all');

  const filteredDocuments = documents.filter((doc) => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterType === 'all' || doc.storage_type === filterType;
    return matchesSearch && matchesFilter;
  });

  const permanentCount = documents.filter(d => d.storage_type === 'permanent').length;
  const temporaryCount = documents.filter(d => d.storage_type === 'temporary').length;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Documents</h2>
        <button
          onClick={onRefresh}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Refresh
        </button>
      </div>

      {/* Search and Filter */}
      <div className="space-y-3 mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm border border-border rounded-md bg-background"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setFilterType('all')}
            className={cn(
              "flex-1 px-3 py-1.5 text-xs rounded-md transition-colors",
              filterType === 'all'
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            All ({documents.length})
          </button>
          <button
            onClick={() => setFilterType('permanent')}
            className={cn(
              "flex-1 px-3 py-1.5 text-xs rounded-md transition-colors flex items-center justify-center gap-1",
              filterType === 'permanent'
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            <HardDrive className="w-3 h-3" />
            {permanentCount}
          </button>
          <button
            onClick={() => setFilterType('temporary')}
            className={cn(
              "flex-1 px-3 py-1.5 text-xs rounded-md transition-colors flex items-center justify-center gap-1",
              filterType === 'temporary'
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            )}
          >
            <Clock className="w-3 h-3" />
            {temporaryCount}
          </button>
        </div>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {filteredDocuments.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            {searchQuery || filterType !== 'all' 
              ? 'No documents found matching your criteria' 
              : 'No documents uploaded yet'}
          </div>
        ) : (
          filteredDocuments.map((doc) => (
            <div
              key={doc.document_id}
              className="group p-3 border border-border rounded-md hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <FileText className="w-4 h-4 text-muted-foreground mt-0.5" />
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium truncate">{doc.filename}</h4>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      {doc.storage_type === 'permanent' ? (
                        <HardDrive className="w-3 h-3" />
                      ) : (
                        <Clock className="w-3 h-3" />
                      )}
                      {doc.storage_type}
                    </span>
                    <span>{doc.total_chunks} chunks</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Added: {new Date(doc.added_date).toLocaleDateString()}
                  </div>
                </div>
                <button
                  onClick={() => onDeleteDocument(doc.document_id)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-destructive/10 rounded transition-all"
                  title="Delete document"
                >
                  <Trash2 className="w-4 h-4 text-destructive" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Summary Stats */}
      <div className="mt-4 pt-4 border-t border-border">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="text-center">
            <div className="text-2xl font-semibold">{documents.length}</div>
            <div className="text-muted-foreground">Total Documents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold">
              {documents.reduce((sum, doc) => sum + doc.total_chunks, 0)}
            </div>
            <div className="text-muted-foreground">Total Chunks</div>
          </div>
        </div>
      </div>
    </div>
  );
}