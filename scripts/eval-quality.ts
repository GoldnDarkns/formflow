// How well does the "photo too blurry / glare" check separate good photos from bad ones?
// Made-up specimens only: each one records how it was "photographed" (flat, desk, blur, glare, sideways).
// usage: npx tsx scripts/eval-quality.ts [train|test|final]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { rgbaToGray } from '../src/core/preprocess';
import { detectPage, warpQuad } from '../src/core/scan';
import { photoQuality } from '../src/core/quality';
import { DATA } from './funsd';

const split = process.argv[2] ?? 'train';
const docs = JSON.parse(fs.readFileSync(path.join(DATA, 'synth', split, 'truth.json'), 'utf8')).documents as any[];
const by: Record<string, { s: number[]; g: number[]; blurry: number; glary: number }> = {};
for (const d of docs) {
  const { data, info } = await sharp(path.join(DATA, 'synth', split, d.file)).rotate().flatten({ background: '#fff' }).resize(2400, 2400, { fit: 'inside', withoutEnlargement: true }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  let color = new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), W = info.width, H = info.height;
  const quad = detectPage(rgbaToGray(color, W, H));
  if (quad) { const w = warpQuad(color, W, H, quad); color = w.data as any; W = w.width; H = w.height; }
  const q = photoQuality(rgbaToGray(color, W, H));
  const b = (by[d.shot] ??= { s: [], g: [], blurry: 0, glary: 0 });
  b.s.push(q.sharpness); b.g.push(q.glare); if (q.blurry) b.blurry++; if (q.glary) b.glary++;
}
const pc = (a: number[], p: number) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * p))];
console.log(`synth/${split}: photo quality by how the specimen was photographed`);
console.log('shot'.padEnd(10) + 'n'.padStart(4) + '  sharpness 10% / 50% / 90%'.padEnd(30) + '  glare 10% / 50% / 90%'.padEnd(28) + ' flagged blurry  flagged glare');
for (const [k, b] of Object.entries(by)) console.log(k.padEnd(10) + String(b.s.length).padStart(4) + `  ${pc(b.s, 0.1).toFixed(2)} / ${pc(b.s, 0.5).toFixed(2)} / ${pc(b.s, 0.9).toFixed(2)}`.padEnd(30) + `  ${pc(b.g, 0.1).toFixed(3)} / ${pc(b.g, 0.5).toFixed(3)} / ${pc(b.g, 0.9).toFixed(3)}`.padEnd(28) + String(b.blurry).padStart(8) + String(b.glary).padStart(15));
