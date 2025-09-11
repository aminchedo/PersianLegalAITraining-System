/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Environment-specific configuration
  env: {
    NEXT_PUBLIC_API_URL: process.env.NODE_ENV === 'production' 
      ? 'https://persian-legal-ai-backend.railway.app'
      : 'http://localhost:8000',
    NEXT_PUBLIC_VERCEL_ENV: process.env.VERCEL_ENV || 'development'
  },
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NODE_ENV === 'production' 
          ? 'https://persian-legal-ai-backend.railway.app/api/:path*'
          : 'http://localhost:8000/api/:path*'
      }
    ]
  },
  
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          }
        ],
      },
      {
        source: '/api/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET, POST, PUT, DELETE, OPTIONS',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'Content-Type, Authorization, X-Requested-With',
          }
        ],
      }
    ]
  },
  
  // Vercel-optimized build settings
  output: 'standalone',
  distDir: '.next',
  generateEtags: false,
  compress: true,
  poweredByHeader: false,
  trailingSlash: false,
  
  // Performance optimizations
  experimental: {
    optimizePackageImports: ['@ant-design/icons', 'lucide-react']
  },
  
  // Image optimization for Vercel
  images: {
    unoptimized: process.env.VERCEL ? true : false,
    domains: ['persian-legal-ai-backend.railway.app']
  },
  
  // Bundle analyzer in development
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Optimize bundle size
    if (!dev && !isServer) {
      config.optimization.splitChunks.chunks = 'all'
      config.optimization.splitChunks.cacheGroups = {
        default: false,
        vendors: false,
        vendor: {
          name: 'vendor',
          chunks: 'all',
          test: /node_modules/
        },
        common: {
          name: 'common',
          minChunks: 2,
          priority: 10,
          reuseExistingChunk: true,
          enforce: true
        }
      }
    }
    
    return config
  }
}

module.exports = nextConfig