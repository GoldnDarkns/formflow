import path from 'node:path'; import fs from 'node:fs';
import { RawImage } from '@huggingface/transformers';
import { loadVisionCritic } from '../src/core/critic-vision';
import { DATA } from './funsd';
const c = await loadVisionCritic('cpu', undefined, path.join(DATA, 'models'));
const dirs = process.argv.slice(2);
for (const d of dirs.length ? dirs : ['public/samples']) for (const f of fs.readdirSync(d).filter((f) => /png|jpg/.test(f)).slice(0, 12)) {
  const t = Date.now(); const o = await c.look((await RawImage.read(path.join(d, f))).rgb());
  console.log(f.padEnd(34), o.type.padEnd(16), (o.confidence * 100).toFixed(0) + '%', Date.now() - t, 'ms');
}
