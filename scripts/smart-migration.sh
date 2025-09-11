#!/bin/bash
# smart-migration.sh - Smart Migration System for Bolt Integration

echo "🔄 Starting smart migration system..."

# Create backup before any changes
echo "💾 Creating backup..."
cd /workspace
git add -A
git commit -m "chore: pre-integration backup" || echo "No changes to commit"

# Initialize migration log
echo "📝 Initializing migration log..."
cat > migration.log << EOF
Migration Log - $(date)
========================
EOF

# Smart component migration
migrate_components_smart() {
    echo "🧠 Smart component migration..."
    
    if [ ! -d "bolt/src/components" ]; then
        echo "⚠️  No bolt components found"
        return 0
    fi
    
    # Create structure
    mkdir -p frontend/src/components/bolt/{components,pages,layouts,hooks}
    
    # Migrate with analysis
    find bolt/src/components -name "*.tsx" -o -name "*.ts" | while read source_file; do
        relative_path=$(echo "$source_file" | sed 's|bolt/src/components/||')
        
        # Determine target location based on file type
        if echo "$relative_path" | grep -q -i "page\|route\|screen"; then
            target_file="frontend/src/components/bolt/pages/$relative_path"
            echo "📄 Migrating page component: $relative_path"
        elif echo "$relative_path" | grep -q "layout"; then
            target_file="frontend/src/components/bolt/layouts/$relative_path"
            echo "🏗️  Migrating layout component: $relative_path"
        else
            target_file="frontend/src/components/bolt/components/$relative_path"
            echo "🧩 Migrating component: $relative_path"
        fi
        
        target_dir=$(dirname "$target_file")
        mkdir -p "$target_dir"
        
        # Copy and transform
        cp "$source_file" "$target_file"
        
        # Transform imports with sed (handle both single and double quotes)
        sed -i.bak 's|from ["\x27]@/components|from "../../components|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/hooks|from "../../../hooks|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/api|from "../../../api|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/types|from "../../../types|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/utils|from "../../../services/boltUtils|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/lib|from "../../../lib|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/context|from "../../../services|g' "$target_file"
        
        # Transform relative imports within bolt components
        sed -i.bak 's|from ["\x27]\./|from "./|g' "$target_file"
        sed -i.bak 's|from ["\x27]\.\./|from "../|g' "$target_file"
        
        # Remove backup files
        rm "$target_file.bak" 2>/dev/null || true
        
        echo "✅ Migrated: $source_file -> $target_file" >> migration.log
    done
}

# Smart hooks migration
migrate_hooks_smart() {
    echo "🪝 Smart hooks migration..."
    
    if [ ! -d "bolt/src/hooks" ]; then
        echo "⚠️  No bolt hooks found"
        return 0
    fi
    
    find bolt/src/hooks -name "*.ts" -o -name "*.tsx" | while read source_file; do
        relative_path=$(echo "$source_file" | sed 's|bolt/src/hooks/||')
        
        # Check if hook already exists in frontend
        frontend_hook="frontend/src/hooks/$relative_path"
        if [ -f "$frontend_hook" ]; then
            # Create bolt-specific version
            target_file="frontend/src/components/bolt/hooks/$relative_path"
            echo "🔄 Creating bolt-specific hook: $relative_path"
        else
            # Move to main hooks directory
            target_file="frontend/src/hooks/$relative_path"
            echo "🪝 Migrating hook: $relative_path"
        fi
        
        target_dir=$(dirname "$target_file")
        mkdir -p "$target_dir"
        
        # Copy and transform
        cp "$source_file" "$target_file"
        
        # Transform imports
        sed -i.bak 's|from ["\x27]@/api|from "../api|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/types|from "../types|g' "$target_file"
        sed -i.bak 's|from ["\x27]@/utils|from "../services/boltUtils|g' "$target_file"
        
        # Remove backup files
        rm "$target_file.bak" 2>/dev/null || true
        
        echo "✅ Migrated hook: $source_file -> $target_file" >> migration.log
    done
}

# Smart API migration
migrate_api_smart() {
    echo "📡 Smart API migration..."
    
    if [ ! -d "bolt/src/api" ]; then
        echo "⚠️  No bolt API found"
        return 0
    fi
    
    # Create consolidated API file
    cat > frontend/src/api/boltApi.ts << 'EOF'
import axios, { AxiosResponse, AxiosError } from 'axios';

interface BoltApiConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
}

class BoltApiService {
  private axiosInstance;
  private config: BoltApiConfig;

  constructor() {
    this.config = {
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
      timeout: 15000,
      retryAttempts: 3
    };

    this.axiosInstance = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        config.headers['X-Request-ID'] = this.generateRequestId();
        return config;
      },
      (error) => {
        console.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor with retry logic
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
          return Promise.reject(error);
        }

        // Retry logic for server errors
        if (error.response?.status >= 500 && originalRequest._retryCount < this.config.retryAttempts) {
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
          await this.delay(1000 * originalRequest._retryCount);
          return this.axiosInstance(originalRequest);
        }

        return Promise.reject(error);
      }
    );
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Health check method
  async healthCheck(): Promise<AxiosResponse> {
    return this.axiosInstance.get('/bolt/health');
  }

EOF

    # Append methods from bolt API files
    find bolt/src/api -name "*.ts" | while read api_file; do
        echo "// Methods from $(basename $api_file)" >> frontend/src/api/boltApi.ts
        echo "" >> frontend/src/api/boltApi.ts
        
        # Extract and transform methods (simplified approach)
        grep -A 10 "static.*async\|async.*(" "$api_file" | while IFS= read -r line; do
            # Transform the line for the new API structure
            if echo "$line" | grep -q "static.*async"; then
                # Convert static methods to instance methods
                transformed_line=$(echo "$line" | sed 's/static async/async/g' | sed 's/this\.request/this.axiosInstance/g')
                echo "  $transformed_line" >> frontend/src/api/boltApi.ts
            elif echo "$line" | grep -q "async.*(" && ! echo "$line" | grep -q "static"; then
                echo "  $line" >> frontend/src/api/boltApi.ts
            elif echo "$line" | grep -q "return.*request"; then
                # Transform request calls
                transformed_line=$(echo "$line" | sed 's/this\.request/this.axiosInstance/g')
                echo "    $transformed_line" >> frontend/src/api/boltApi.ts
            elif [ -n "$line" ] && ! echo "$line" | grep -q "^--$"; then
                echo "    $line" >> frontend/src/api/boltApi.ts
            fi
        done
        
        echo "" >> frontend/src/api/boltApi.ts
    done
    
    # Close the class
    cat >> frontend/src/api/boltApi.ts << 'EOF'
}

export const boltApi = new BoltApiService();
export default boltApi;
EOF

    echo "✅ API consolidated: frontend/src/api/boltApi.ts" >> migration.log
}

# Create Bolt types file
create_bolt_types() {
    echo "📝 Creating Bolt types..."
    
    cat > frontend/src/types/bolt.ts << 'EOF'
// Bolt-specific type definitions
export namespace Bolt {
  export interface TrainingSession {
    id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    modelId: string;
    config: TrainingConfig;
    startTime?: Date;
    endTime?: Date;
    metrics?: TrainingMetrics;
  }

  export interface TrainingConfig {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    rank: number;
    alpha: number;
    target_modules: string[];
    quantization_bits: number;
    adaptive_rank: boolean;
  }

  export interface TrainingMetrics {
    loss: number;
    accuracy: number;
    cpu: number;
    memory: number;
    gpu: number;
  }

  export interface Model {
    id: string;
    name: string;
    status: 'active' | 'training' | 'inactive';
    accuracy: number;
    lastTrained: Date;
  }

  export interface Document {
    id: string;
    name: string;
    type: string;
    size: number;
    status: 'uploaded' | 'processing' | 'processed' | 'failed';
    uploadDate: Date;
    processedDate?: Date;
  }

  export interface Analytics {
    totalDocuments: number;
    processedDocuments: number;
    successRate: number;
    averageProcessingTime: number;
  }
}
EOF

    echo "✅ Created Bolt types: frontend/src/types/bolt.ts" >> migration.log
}

# Create Bolt context
create_bolt_context() {
    echo "🔄 Creating Bolt context..."
    
    cat > frontend/src/services/boltContext.tsx << 'EOF'
import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { Bolt } from '../types/bolt';

interface BoltState {
  trainingSessions: Bolt.TrainingSession[];
  models: Bolt.Model[];
  documents: Bolt.Document[];
  analytics: Bolt.Analytics | null;
  isLoading: boolean;
  error: string | null;
}

type BoltAction = 
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_TRAINING_SESSIONS'; payload: Bolt.TrainingSession[] }
  | { type: 'SET_MODELS'; payload: Bolt.Model[] }
  | { type: 'SET_DOCUMENTS'; payload: Bolt.Document[] }
  | { type: 'SET_ANALYTICS'; payload: Bolt.Analytics }
  | { type: 'UPDATE_TRAINING_SESSION'; payload: { id: string; updates: Partial<Bolt.TrainingSession> } };

const initialState: BoltState = {
  trainingSessions: [],
  models: [],
  documents: [],
  analytics: null,
  isLoading: false,
  error: null,
};

function boltReducer(state: BoltState, action: BoltAction): BoltState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_TRAINING_SESSIONS':
      return { ...state, trainingSessions: action.payload };
    case 'SET_MODELS':
      return { ...state, models: action.payload };
    case 'SET_DOCUMENTS':
      return { ...state, documents: action.payload };
    case 'SET_ANALYTICS':
      return { ...state, analytics: action.payload };
    case 'UPDATE_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: state.trainingSessions.map(session =>
          session.id === action.payload.id 
            ? { ...session, ...action.payload.updates }
            : session
        )
      };
    default:
      return state;
  }
}

const BoltContext = createContext<{
  state: BoltState;
  dispatch: React.Dispatch<BoltAction>;
} | null>(null);

export const BoltProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(boltReducer, initialState);

  return (
    <BoltContext.Provider value={{ state, dispatch }}>
      {children}
    </BoltContext.Provider>
  );
};

export const useBolt = () => {
  const context = useContext(BoltContext);
  if (!context) {
    throw new Error('useBolt must be used within a BoltProvider');
  }
  return context;
};
EOF

    echo "✅ Created Bolt context: frontend/src/services/boltContext.tsx" >> migration.log
}

# Execute migrations
echo "🚀 Starting migration process..."

migrate_components_smart
migrate_hooks_smart  
migrate_api_smart
create_bolt_types
create_bolt_context

echo "📊 Migration completed. Check migration.log for details."

# Summary
echo ""
echo "📋 MIGRATION SUMMARY:"
echo "===================="
echo "✅ Components migrated to: frontend/src/components/bolt/"
echo "✅ Hooks migrated to: frontend/src/hooks/ and frontend/src/components/bolt/hooks/"
echo "✅ API consolidated to: frontend/src/api/boltApi.ts"
echo "✅ Types created: frontend/src/types/bolt.ts"
echo "✅ Context created: frontend/src/services/boltContext.tsx"
echo ""
echo "🔍 Next steps:"
echo "1. Review migrated files for import errors"
echo "2. Update dashboard routes to include Bolt components"
echo "3. Run tests to validate migration"
echo ""