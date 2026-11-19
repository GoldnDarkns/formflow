// Embed the critic's type descriptions with MobileCLIP-S0's text encoder (fp32)
// and test the vision critic on the samples and the FUNSD test forms.
// usage: npx tsx scripts/critic-embed.ts
import path from 'node:path'; import fs from 'node:fs';
import { AutoTokenizer, CLIPTextModelWithProjection, RawImage, env } from '@huggingface/transformers';
import { CRITIC_ID, PROMPTS, loadVisionCritic, type TextEmbeds } from '../src/core/critic-vision';
import { DATA } from './funsd';
env.localModelPath = path.join(DATA, 'models'); env.allowRemoteModels = false;
const tok = await AutoTokenizer.from_pretrained(CRITIC_ID);
const model = await CLIPTextModelWithProjection.from_pretrained(CRITIC_ID, { dtype: 'fp32' } as any);
const embeds: number[][] = [];
for (const [, ps] of PROMPTS) {
  const t = await model(tok(ps, { padding: 'max_length', truncation: true }));
  const D = t.text_embeds.dims[1], d = t.text_embeds.data as Float32Array;
  const mean = new Array(D).fill(0);
  ps.forEach((_, i) => { const v = d.slice(i * D, i * D + D), n = Math.hypot(...v); v.forEach((x, k) => (mean[k] += x / n / ps.length)); });
  const n = Math.hypot(...mean);
  embeds.push(mean.map((x) => +(x / n).toFixed(5)));
}
const text: TextEmbeds = { types: PROMPTS.map((p) => p[0]), dim: embeds[0].length, embeds };
fs.writeFileSync('public/models/critic-text.json', JSON.stringify(text));
const c = await loadVisionCritic('cpu', text, undefined, path.join(DATA, 'models'));
const expect: Record<string, string> = { 'specimen-driving-licence.jpg': 'driving_licence', 'specimen-emirates-id-back.jpg': 'emirates_id', 'specimen-emirates-id-front.jpg': 'emirates_id', 'specimen-passport.jpg': 'passport', 'specimen-visa.jpg': 'visa', 'form-fax-cover.png': 'form', 'form-report.png': 'form' };
let ok = 0;
for (const [f, e] of Object.entries(expect)) { const o = await c.look((await RawImage.read('public/samples/' + f)).rgb()); ok += +(o.type === e); console.log(f.padEnd(34), o.type.padEnd(16), (o.confidence * 100).toFixed(0) + '%', o.type === e ? '' : 'WRONG'); }
const dir = path.join(DATA, 'funsd/dataset/testing_data/images');
let forms = 0, n = 0, ms = 0;
for (const f of fs.readdirSync(dir)) { const t = Date.now(); const o = await c.look((await RawImage.read(path.join(dir, f))).rgb()); ms += Date.now() - t; n++; forms += +(o.type === 'form'); }
const res = { specimens: { correct: ok, total: 7 }, funsd_forms: { correct: forms, total: n }, msPerImage: ms / n };
console.log(res);
fs.writeFileSync(path.join(DATA, 'critic.json'), JSON.stringify(res));
