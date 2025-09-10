const { BasicTestRunner } = require('./tests/basic-test-suite.js');

async function runTests() {
    const runner = new BasicTestRunner();
    
    try {
        console.log('🚀 Executing comprehensive test suite...');
        const results = await runner.runAllTests();
        
        console.log('\n📊 TEST RESULTS SUMMARY:');
        console.log(`Total Suites: ${results.summary.totalSuites}`);
        console.log(`Total Tests: ${results.summary.totalTests}`);
        console.log(`Passed: ${results.summary.passedTests} ✅`);
        console.log(`Failed: ${results.summary.failedTests} ❌`);
        console.log(`Success Rate: ${results.summary.successRate.toFixed(2)}%`);
        
        // Generate detailed report
        const report = runner.generateReport(results);
        
        // Write report to file
        const fs = require('fs');
        fs.writeFileSync('test-reports/comprehensive_test_report_' + new Date().toISOString().slice(0,19).replace(/:/g, '') + '.md', report);
        
        console.log('\n📄 Detailed report generated in test-reports/');
        
        // Exit with appropriate code
        process.exit(results.summary.failedTests > 0 ? 1 : 0);
        
    } catch (error) {
        console.error('❌ Test suite execution failed:', error);
        process.exit(1);
    }
}

runTests();
