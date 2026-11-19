// Run Florence-2 OCR_WITH_REGION over FUNSD test pages and cache the regions.
// usage: npx tsx scripts/florence-funsd.ts <tilesCols>x<tilesRows> [limit]
import fs from 'node:fs';
import path from 'node:path';
import { RawImage } from '@huggingface/transformers';
import { loadFlorence } from '../src/core/florence';
import { DATA, pages } from './funsd';

const [tileArg = '1x1', limitArg] = process.argv.slice(2);
const [cols, rows] = tileArg.split('x').map(Number);
const outDir = path.join(DATA, 'florence', `testing-${tileArg}`);
fs.mkdirSync(outDir, { recursive: true });

const reader = await loadFlorence('cpu', undefined, path.join(DATA, 'models'));
const files = pages('testing').slice(0, limitArg ? +limitArg : undefined);
for (const f of files) {
  const out = path.join(outDir, f);
  if (fs.existsSync(out)) continue;
  const img = await RawImage.read(path.join(DATA, 'funsd/dataset/testing_data/images', f.replace('.json', '.png')));
  const t = Date.now();
  const regions = await reader.read(img.rgb(), { cols, rows });
  fs.writeFileSync(out, JSON.stringify({ ms: Date.now() - t, regions }));
  console.log(f, regions.length, 'regions', Date.now() - t, 'ms', regions.slice(0, 4).map((r) => r.text).join(' | '));
}
