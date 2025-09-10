/**
 * Persian Loader Component
 * کامپوننت بارگذاری فارسی با انیمیشن‌های زیبا
 */

import React from 'react';
import { Brain, Zap, Database, Activity } from 'lucide-react';

interface PersianLoaderProps {
  type?: 'spinner' | 'dots' | 'pulse' | 'wave' | 'brain' | 'text' | 'skeleton';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  message?: string;
  progress?: number; // 0-100
  showProgress?: boolean;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  fullScreen?: boolean;
  transparent?: boolean;
}

const PersianLoader: React.FC<PersianLoaderProps> = ({
  type = 'spinner',
  size = 'md',
  message = 'در حال بارگذاری...',
  progress,
  showProgress = false,
  color = 'primary',
  fullScreen = false,
  transparent = false
}) => {
  // Size configurations
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  const textSizes = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl'
  };

  // Color configurations
  const colorClasses = {
    primary: 'text-blue-600',
    secondary: 'text-green-600',
    success: 'text-emerald-600',
    warning: 'text-amber-600',
    error: 'text-red-600'
  };

  const bgColorClasses = {
    primary: 'bg-blue-600',
    secondary: 'bg-green-600',
    success: 'bg-emerald-600',
    warning: 'bg-amber-600',
    error: 'bg-red-600'
  };

  // Spinner Component
  const SpinnerLoader = () => (
    <div className={`animate-spin rounded-full border-2 border-gray-200 ${sizeClasses[size]}`}>
      <div className={`rounded-full border-2 border-transparent border-t-current ${colorClasses[color]}`}></div>
    </div>
  );

  // Dots Component
  const DotsLoader = () => (
    <div className="flex space-x-1 rtl:space-x-reverse">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={`rounded-full ${bgColorClasses[color]} animate-pulse`}
          style={{
            width: size === 'sm' ? '4px' : size === 'md' ? '6px' : size === 'lg' ? '8px' : '10px',
            height: size === 'sm' ? '4px' : size === 'md' ? '6px' : size === 'lg' ? '8px' : '10px',
            animationDelay: `${i * 0.2}s`,
            animationDuration: '1s'
          }}
        ></div>
      ))}
    </div>
  );

  // Pulse Component
  const PulseLoader = () => (
    <div className={`rounded-full ${bgColorClasses[color]} animate-ping ${sizeClasses[size]}`}></div>
  );

  // Wave Component
  const WaveLoader = () => (
    <div className="flex items-end space-x-1 rtl:space-x-reverse">
      {[0, 1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className={`${bgColorClasses[color]} animate-pulse`}
          style={{
            width: size === 'sm' ? '2px' : size === 'md' ? '3px' : size === 'lg' ? '4px' : '5px',
            height: size === 'sm' ? '12px' : size === 'md' ? '16px' : size === 'lg' ? '20px' : '24px',
            animationDelay: `${i * 0.1}s`,
            animationDuration: '0.8s'
          }}
        ></div>
      ))}
    </div>
  );

  // Brain Component (AI-themed)
  const BrainLoader = () => (
    <div className={`${colorClasses[color]} animate-pulse`}>
      <Brain className={sizeClasses[size]} />
      <div className="absolute inset-0 animate-ping">
        <Brain className={`${sizeClasses[size]} opacity-20`} />
      </div>
    </div>
  );

  // Text Component (Persian typing effect)
  const TextLoader = () => {
    const [displayText, setDisplayText] = React.useState('');
    const fullText = message;

    React.useEffect(() => {
      let currentIndex = 0;
      const interval = setInterval(() => {
        if (currentIndex <= fullText.length) {
          setDisplayText(fullText.slice(0, currentIndex));
          currentIndex++;
        } else {
          currentIndex = 0;
        }
      }, 100);

      return () => clearInterval(interval);
    }, [fullText]);

    return (
      <div className={`${textSizes[size]} ${colorClasses[color]} font-medium`}>
        {displayText}
        <span className="animate-blink">|</span>
      </div>
    );
  };

  // Skeleton Component
  const SkeletonLoader = () => (
    <div className="animate-pulse space-y-3">
      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
      <div className="h-4 bg-gray-200 rounded w-1/2"></div>
      <div className="h-4 bg-gray-200 rounded w-5/6"></div>
    </div>
  );

  // Progress Bar Component
  const ProgressBar = () => (
    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
      <div
        className={`h-2 ${bgColorClasses[color]} transition-all duration-300 ease-out`}
        style={{ 
          width: `${Math.min(Math.max(progress || 0, 0), 100)}%`,
          background: `linear-gradient(90deg, ${bgColorClasses[color]} 0%, ${bgColorClasses[color]}aa 100%)`
        }}
      ></div>
    </div>
  );

  // Render appropriate loader
  const renderLoader = () => {
    switch (type) {
      case 'dots':
        return <DotsLoader />;
      case 'pulse':
        return <PulseLoader />;
      case 'wave':
        return <WaveLoader />;
      case 'brain':
        return <BrainLoader />;
      case 'text':
        return <TextLoader />;
      case 'skeleton':
        return <SkeletonLoader />;
      default:
        return <SpinnerLoader />;
    }
  };

  // Container classes
  const containerClasses = fullScreen
    ? `fixed inset-0 z-50 flex items-center justify-center ${
        transparent ? 'bg-black bg-opacity-20' : 'bg-white'
      }`
    : 'flex items-center justify-center p-4';

  return (
    <div className={containerClasses} dir="rtl">
      <div className="flex flex-col items-center space-y-4">
        {/* Loader Animation */}
        {type !== 'text' && type !== 'skeleton' && (
          <div className="relative">
            {renderLoader()}
          </div>
        )}

        {/* Text Message */}
        {type !== 'text' && message && (
          <div className={`${textSizes[size]} ${colorClasses[color]} font-medium text-center`}>
            {message}
          </div>
        )}

        {/* Text Loader */}
        {type === 'text' && <TextLoader />}

        {/* Skeleton Loader */}
        {type === 'skeleton' && <SkeletonLoader />}

        {/* Progress Bar */}
        {showProgress && typeof progress === 'number' && (
          <div className="w-full max-w-xs">
            <ProgressBar />
            <div className="flex justify-between items-center mt-1 text-xs text-gray-500">
              <span>۰٪</span>
              <span className={colorClasses[color]}>
                {progress.toLocaleString('fa-IR')}٪
              </span>
              <span>۱۰۰٪</span>
            </div>
          </div>
        )}

        {/* Additional Info */}
        {fullScreen && (
          <div className="text-xs text-gray-400 text-center mt-4">
            لطفاً صبر کنید...
          </div>
        )}
      </div>
    </div>
  );
};

// Preset Loaders for common use cases
export const AITrainingLoader: React.FC<{ progress?: number }> = ({ progress }) => (
  <PersianLoader
    type="brain"
    size="lg"
    message="در حال آموزش مدل هوش مصنوعی..."
    progress={progress}
    showProgress={typeof progress === 'number'}
    color="primary"
    fullScreen
  />
);

export const DataProcessingLoader: React.FC = () => (
  <PersianLoader
    type="wave"
    size="md"
    message="در حال پردازش داده‌ها..."
    color="secondary"
  />
);

export const SystemHealthLoader: React.FC = () => (
  <PersianLoader
    type="pulse"
    size="sm"
    message="بررسی وضعیت سیستم..."
    color="success"
  />
);

export const DocumentAnalysisLoader: React.FC = () => (
  <PersianLoader
    type="dots"
    size="md"
    message="تحلیل اسناد حقوقی..."
    color="primary"
  />
);

export const PageSkeletonLoader: React.FC = () => (
  <PersianLoader
    type="skeleton"
    size="md"
  />
);

// CSS for custom animations (to be added to global styles)
export const loaderStyles = `
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.animate-blink {
  animation: blink 1s infinite;
}

.loader-gradient {
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.4),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}
`;

export default PersianLoader;