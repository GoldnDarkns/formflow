// Evaluate the pipeline on your own labelled documents, per subgroup.
// Layout: <dir>/<subgroup>/<name>.(jpg|png) + <name>.labels.json (exported from
// the app with "Labelled example" after correcting the fields).
// usage: npx tsx scripts/eval-own.ts ../data/own
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import * as ortNode from 'onnxruntime-node';
import { createWorker, PSM } from 'tesseract.js';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale, downscale } from '../src/core/preprocess';
import { wordsFromTesseract, cleanWords, similarity, norm } from '../src/core/ocr';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';

const dir = path.resolve(process.argv[2] ?? '../data/own');
const ROOT = path.resolve(import.meta.dirname, '..');
const clf = await loadClassifiers(ortNode as any, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));
const tess = await createWorker('eng', 1, { langPath: path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int'), cacheMethod: 'none' });
await tess.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });

const clean = (s: string) => norm(s).replace(/[^\p{L}\p{N} ]/gu, '').replace(/\s+/g, ' ').trim();
const sim = (a: string, b: string) => similarity(clean(a), clean(b));

const groups = new Map<string, { tp: number; nPred: number; nGold: number; ms: number; n: number; valueExact: number }>();
const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
for (const img of walk(dir).filter((f) => /\.(png|jpe?g|webp)$/i.test(f))) {
  const lab = img.replace(/\.[^.]+$/, '.labels.json');
  if (!fs.existsSync(lab)) { console.warn('no labels for', img); continue; }
  const L = JSON.parse(fs.readFileSync(lab, 'utf8'));
  const byId = new Map<number, any>(L.form.map((e: any) => [e.id, e]));
  const gold: { key: string; value: string }[] = [];
  for (const e of L.form)
    if (e.label === 'question') {
      const as = e.linking.filter(([q]: number[]) => q === e.id).map(([, a]: number[]) => byId.get(a)?.text).filter(Boolean);
      if (as.length) gold.push({ key: e.text.replace(/[:\s.]+$/, ''), value: as.join(' ') });
    }
  const t = Date.now();
  const { data, info } = await sharp(img).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  let g = rgbaToGray(data, info.width, info.height);
  g = downscale(g, 2400).img;
  const W = g.width, H = g.height;
  const scale = ocrScale(W, H);
  const pre = preprocess(resize(g, scale)).img;
  const png = await sharp(Buffer.from(grayToRgba(pre)), { raw: { width: pre.width, height: pre.height, channels: 4 } }).png().toBuffer();
  const { data: r } = await tess.recognize(png, {}, { blocks: true });
  const words = cleanWords(wordsFromTesseract(r.blocks as any, scale));
  const hasMrz = words.some((w) => (w.text.match(/</g)?.length ?? 0) >= 3);
  const kind = /passport|visa|licen[cs]e|id/i.test(img) || hasMrz ? 'id' : 'form';
  const an = await analyse(words, W, H, clf, undefined, kind);
  const P = an.fields.filter((f) => f.value);
  const used = new Set<number>();
  let tp = 0, exact = 0;
  for (const f of P) {
    let best = -1, bs = 0;
    gold.forEach((gf, j) => {
      if (used.has(j)) return;
      const ks = sim(f.key, gf.key), vs = sim(f.value, gf.value);
      if (ks >= 0.8 && vs >= 0.8 && ks + vs > bs) { bs = ks + vs; best = j; }
    });
    if (best >= 0) { used.add(best); tp++; if (clean(f.value) === clean(gold[best].value)) exact++; }
  }
  const grp = path.relative(dir, path.dirname(img)) || 'all';
  const G = groups.get(grp) ?? { tp: 0, nPred: 0, nGold: 0, ms: 0, n: 0, valueExact: 0 };
  G.tp += tp; G.nPred += P.length; G.nGold += gold.length; G.ms += Date.now() - t; G.n++; G.valueExact += exact;
  groups.set(grp, G);
  console.log(path.relative(dir, img), `${tp}/${gold.length} fields`, kind);
}
const out: any = {};
for (const [k, G] of groups) {
  const p = G.tp / Math.max(1, G.nPred), r = G.tp / Math.max(1, G.nGold);
  out[k] = { documents: G.n, precision: p, recall: r, f1: (2 * p * r) / Math.max(1e-9, p + r), exactValue: G.valueExact / Math.max(1, G.tp), msPerDoc: G.ms / G.n };
}
console.table(out);
fs.writeFileSync(path.join(ROOT, 'public/eval/own.json'), JSON.stringify({ generated: new Date().toISOString(), groups: out }, null, 1));
await tess.terminate();
