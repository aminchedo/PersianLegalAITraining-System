// dependency-analyzer.ts
import fs from 'fs';
import path from 'path';

interface DependencyAnalysis {
  conflicting: string[];
  missing: string[];
  compatible: string[];
  versionMismatches: Array<{package: string, bolt: string, main: string}>;
}

export class DependencyAnalyzer {
  static async analyzeDependencies(): Promise<DependencyAnalysis> {
    console.log('📊 Analyzing dependencies...');
    
    const boltPackage = JSON.parse(fs.readFileSync('bolt/package.json', 'utf8'));
    const mainPackage = JSON.parse(fs.readFileSync('frontend/package.json', 'utf8'));
    
    const analysis: DependencyAnalysis = {
      conflicting: [],
      missing: [],
      compatible: [],
      versionMismatches: []
    };

    const boltDeps = { ...boltPackage.dependencies, ...boltPackage.devDependencies };
    const mainDeps = { ...mainPackage.dependencies, ...mainPackage.devDependencies };

    console.log('Bolt dependencies:', Object.keys(boltDeps).length);
    console.log('Frontend dependencies:', Object.keys(mainDeps).length);

    for (const [pkg, version] of Object.entries(boltDeps)) {
      if (mainDeps[pkg]) {
        if (mainDeps[pkg] !== version) {
          analysis.versionMismatches.push({
            package: pkg,
            bolt: version as string,
            main: mainDeps[pkg]
          });
        } else {
          analysis.compatible.push(pkg);
        }
      } else {
        analysis.missing.push(pkg);
      }
    }

    console.log('📊 Dependency Analysis Results:');
    console.log(`✅ Compatible: ${analysis.compatible.length}`);
    console.log(`⚠️  Version mismatches: ${analysis.versionMismatches.length}`);
    console.log(`📦 Missing: ${analysis.missing.length}`);

    // Print details
    if (analysis.versionMismatches.length > 0) {
      console.log('\n⚠️  Version Mismatches:');
      analysis.versionMismatches.forEach(mismatch => {
        console.log(`   ${mismatch.package}: Bolt(${mismatch.bolt}) vs Frontend(${mismatch.main})`);
      });
    }

    if (analysis.missing.length > 0) {
      console.log('\n📦 Missing Dependencies:');
      analysis.missing.forEach(pkg => {
        console.log(`   ${pkg}: ${boltDeps[pkg]}`);
      });
    }

    return analysis;
  }

  static async resolveDependencies(analysis: DependencyAnalysis): Promise<void> {
    console.log('🔧 Resolving dependencies...');
    
    // Create updated package.json
    const mainPackage = JSON.parse(fs.readFileSync('frontend/package.json', 'utf8'));
    const boltPackage = JSON.parse(fs.readFileSync('bolt/package.json', 'utf8'));
    
    // Add missing dependencies
    for (const pkg of analysis.missing) {
      const version = boltPackage.dependencies[pkg] || boltPackage.devDependencies[pkg];
      if (boltPackage.dependencies[pkg]) {
        mainPackage.dependencies[pkg] = version;
        console.log(`➕ Added dependency: ${pkg}@${version}`);
      } else if (boltPackage.devDependencies[pkg]) {
        mainPackage.devDependencies[pkg] = version;
        console.log(`➕ Added devDependency: ${pkg}@${version}`);
      }
    }

    // Handle version mismatches (use higher version)
    for (const mismatch of analysis.versionMismatches) {
      const boltVersion = mismatch.bolt.replace(/[\^~]/, '');
      const mainVersion = mismatch.main.replace(/[\^~]/, '');
      
      // Simple version comparison (this could be more sophisticated)
      const useVersion = this.compareVersions(boltVersion, mainVersion) > 0 ? mismatch.bolt : mismatch.main;
      
      if (mainPackage.dependencies[mismatch.package]) {
        mainPackage.dependencies[mismatch.package] = useVersion;
      } else if (mainPackage.devDependencies[mismatch.package]) {
        mainPackage.devDependencies[mismatch.package] = useVersion;
      }
      
      console.log(`🔄 Updated ${mismatch.package} to ${useVersion}`);
    }

    // Write updated package.json
    fs.writeFileSync('frontend/package.json', JSON.stringify(mainPackage, null, 2));
    console.log('✅ Updated frontend/package.json');
  }

  private static compareVersions(a: string, b: string): number {
    const aParts = a.split('.').map(n => parseInt(n));
    const bParts = b.split('.').map(n => parseInt(n));
    
    for (let i = 0; i < Math.max(aParts.length, bParts.length); i++) {
      const aPart = aParts[i] || 0;
      const bPart = bParts[i] || 0;
      
      if (aPart > bPart) return 1;
      if (aPart < bPart) return -1;
    }
    
    return 0;
  }
}

// Main execution
async function main() {
  try {
    const analysis = await DependencyAnalyzer.analyzeDependencies();
    await DependencyAnalyzer.resolveDependencies(analysis);
    
    // Generate installation commands
    console.log('\n🚀 Next steps:');
    console.log('cd frontend && npm install');
    console.log('# This will install all missing dependencies');
    
  } catch (error) {
    console.error('❌ Error analyzing dependencies:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}