// Read a private document set using every processor core, then print one report.
// usage: node scripts/eval-fast.mjs <train|test|final> [own|synth] [copies=8]
import { spawn } from 'node:child_process';
const [split = 'test', root = 'own', copies = '8'] = process.argv.slice(2);
const n = +copies;
const run = (env) => new Promise((res) => {
  const p = spawn(process.execPath, ['node_modules/tsx/dist/cli.mjs', 'scripts/eval-real.ts', split, root], { env: { ...process.env, ...env }, stdio: env.REUSE ? 'inherit' : 'ignore' });
  p.on('exit', res);
});
const t = Date.now();
await Promise.all(Array.from({ length: n }, (_, i) => run({ SHARD: `${i}/${n}` })));
console.log(`read with ${n} copies in ${Math.round((Date.now() - t) / 1000)} s`);
await run({ REUSE: '1' });
