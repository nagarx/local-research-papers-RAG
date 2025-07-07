import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  swcMinify: true,
  
  // Disable image optimization for Docker deployment
  images: {
    unoptimized: true,
  },
  
  // API rewrites for development
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL 
          ? `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`
          : 'http://localhost:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
