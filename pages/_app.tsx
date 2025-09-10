import type { AppProps } from 'next/app'
import { AppProvider } from '../src/contexts/AppContext'
import '../src/styles/globals.css'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <AppProvider>
      <Component {...pageProps} />
    </AppProvider>
  )
}