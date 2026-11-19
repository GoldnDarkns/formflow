import path from 'node:path'; import fs from 'node:fs';
import { RawImage } from '@huggingface/transformers';
import { loadFlorence } from '../src/core/florence';
import { DATA } from './funsd';
const r = await loadFlorence('cpu', undefined, path.join(DATA, 'models'));
const out: any = {};
for (const f of fs.readdirSync('public/samples')) {
  const img = (await RawImage.read('public/samples/' + f)).rgb();
  const t = Date.now();
  const a = await r.caption(img, '<CAPTION>'); const t1 = Date.now() - t;
  const b = await r.caption(img, '<DETAILED_CAPTION>');
  out[f] = { a, b }; console.log(f, t1, 'ms |', a, '||', b, Date.now() - t - t1, 'ms');
}
fs.writeFileSync(path.join(DATA, 'captions.json'), JSON.stringify(out, null, 1));
