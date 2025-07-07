'use client';

import React from 'react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { FileText, Database, HardDrive, Clock } from 'lucide-react';
import { AnalyticsOverview } from '@/lib/api';

interface AnalyticsProps {
  data?: AnalyticsOverview;
}

export function Analytics({ data }: AnalyticsProps) {
  if (!data) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading analytics...</p>
      </div>
    );
  }

  const documentData = [
    { name: 'Permanent', value: data.documents.permanent, color: '#3b82f6' },
    { name: 'Temporary', value: data.documents.temporary, color: '#10b981' },
  ];

  const statsCards = [
    {
      title: 'Total Documents',
      value: data.documents.total,
      icon: FileText,
      color: 'text-blue-500',
    },
    {
      title: 'Total Chunks',
      value: data.chunks.total,
      icon: Database,
      color: 'text-green-500',
    },
    {
      title: 'Avg Chunks/Doc',
      value: data.chunks.average_per_doc.toFixed(1),
      icon: BarChart,
      color: 'text-purple-500',
    },
  ];

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {statsCards.map((stat) => (
            <div
              key={stat.title}
              className="p-6 border border-border rounded-lg bg-card"
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.title}</p>
                  <p className="text-3xl font-semibold mt-2">{stat.value}</p>
                </div>
                <stat.icon className={`w-8 h-8 ${stat.color}`} />
              </div>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Document Distribution */}
          <div className="p-6 border border-border rounded-lg bg-card">
            <h3 className="text-lg font-semibold mb-4">Document Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={documentData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {documentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* System Info */}
          <div className="p-6 border border-border rounded-lg bg-card">
            <h3 className="text-lg font-semibold mb-4">System Information</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded">
                <div className="flex items-center gap-3">
                  <HardDrive className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm">Permanent Storage</span>
                </div>
                <span className="text-sm font-medium">{data.documents.permanent} docs</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded">
                <div className="flex items-center gap-3">
                  <Clock className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm">Temporary Storage</span>
                </div>
                <span className="text-sm font-medium">{data.documents.temporary} docs</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded">
                <div className="flex items-center gap-3">
                  <Database className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm">Vector Database</span>
                </div>
                <span className="text-sm font-medium text-green-600">Active</span>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-6 p-4 bg-muted/30 rounded-lg">
          <p className="text-xs text-muted-foreground">
            Last updated: {new Date(data.timestamp).toLocaleString()}
          </p>
        </div>
      </div>
    </div>
  );
}