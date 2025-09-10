/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: process.env.NODE_ENV === 'production' 
          ? 'https://persian-legal-ai-backend.vercel.app/api/:path*'
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
        ],
      },
    ]
  },
  output: 'standalone',
  distDir: '.next',
  generateEtags: false,
  compress: true,
  poweredByHeader: false,
  trailingSlash: false
}

module.exports = nextConfig