#!/usr/bin/env python3
"""
Test runner for Persian Legal AI Training System.

This script runs all tests and provides a comprehensive test report.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_pattern=None, verbose=False, coverage=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.append("--cov=.")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term")
    
    # Add specific test pattern
    if test_pattern:
        cmd.append(f"tests/{test_pattern}")
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return result.returncode

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for Persian Legal AI Training System")
    parser.add_argument("--pattern", "-p", help="Test pattern to run (e.g., test_dora_trainer.py)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--all", "-a", action="store_true", help="Run all tests with coverage and verbose")
    
    args = parser.parse_args()
    
    if args.all:
        args.verbose = True
        args.coverage = True
    
    # Run tests
    exit_code = run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()