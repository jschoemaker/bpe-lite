'use strict';

console.log('bpe-lite test suite');
console.log('═'.repeat(40));
console.log('');

const suites = [
  { name: 'BPE engine',  file: './bpe.test.js' },
  { name: 'Providers',   file: './providers.test.js' },
];

let totalPassed = 0;
let totalFailed = 0;

for (const suite of suites) {
  try {
    const result = require(suite.file);
    totalPassed += result.passed || 0;
    totalFailed += result.failed || 0;
  } catch (err) {
    console.error(`ERROR in ${suite.name}: ${err.message}`);
    totalFailed++;
  }
}

console.log('═'.repeat(40));
console.log(`Total: ${totalPassed} passed, ${totalFailed} failed`);

if (totalFailed > 0) process.exit(1);
