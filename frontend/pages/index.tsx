import React from 'react'
import Head from 'next/head'
import CompletePersianAIDashboard from '../src/components/CompletePersianAIDashboard'

export default function HomePage() {
  return (
    <>
      <Head>
        <title>سامانه هوش مصنوعی حقوقی فارسی</title>
        <meta name="description" content="Persian Legal AI System - Advanced Document Processing and Classification" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <meta charSet="UTF-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
      </Head>
      
      <CompletePersianAIDashboard />
    </>
  )
}