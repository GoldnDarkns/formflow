// Arabic reading on made-up data (scripts/make-arabic.mjs). Prints counts and error rates only.
//   npx tsx scripts/eval-arabic.ts            lines: character error per crop width, Arabic/English decision; cards: name and nationality
//   SYNTH=test npx tsx scripts/eval-arabic.ts also counts segments wrongly taken for Arabic on the English-only specimens
import fs from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';
import { loadPpOcr, loadRecogniser, toReadingOrder, arabicShare, ARABIC_SHARE, ARABIC_CONF, type Recogniser } from '../src/core/ppocr';
import { arabicFacts, editDistance, plainArabic } from '../src/core/facts';
import { rgbaToGray, grayToRgba, resize, ocrScale, preprocess } from '../src/core/preprocess';

const ROOT = path.resolve('.');
const M = path.join(ROOT, 'public/models/ppocr');
const DATA = path.resolve('../data/arabic');
const dict = (f: string) => fs.readFileSync(path.join(M, f), 'utf8');
const pp = await loadPpOcr(ort as any, path.join(M, 'det.onnx'), path.join(M, 'rec.onnx'), dict('dict.txt'), undefined, { maxSide: 1600, boxThresh: 0.55, unclip: 0.8 });
const variants: Record<string, Recogniser> = {};
for (const [name, opt] of [['normal width', { widen: 1 }], ['x1.5', { widen: 1.5 }], ['x2', { widen: 2 }], ['x2, at least 320', { widen: 2, minW: 320 }]] as const)
  variants[name] = await loadRecogniser(ort as any, path.join(M, 'arabic_rec.onnx'), dict('arabic_dict.txt'), undefined, opt);
const png = (f: string) => PNG.sync.read(fs.readFileSync(f));
const cer = (a: string, b: string) => editDistance(a, b) / Math.max(1, b.length);

// ---- lines
const truth: { file: string; text: string; script: 'ar' | 'en' }[] = JSON.parse(fs.readFileSync(path.join(DATA, 'lines/truth.json'), 'utf8'));
const stat: Record<string, { e: number; n: number; exact: number; ms: number }> = {};
const decide: { script: string; share: number; conf: number; v: string }[] = [];
for (const t of truth) {
  const p = png(path.join(DATA, 'lines', t.file));
  const box: [number, number, number, number] = [0, 0, p.width, p.height];
  for (const [name, r] of Object.entries(variants)) {
    const t0 = performance.now();
    const [rec] = await r.recognise(p.data, p.width, [box]);
    const ms = performance.now() - t0;
    const text = toReadingOrder(rec.text);
    if (name === 'x1.5') decide.push({ script: t.script, share: arabicShare(text), conf: rec.conf, v: name });
    if (t.script !== 'ar') continue;
    const s = (stat[name] ??= { e: 0, n: 0, exact: 0, ms: 0 });
    const a = plainArabic(text), b = plainArabic(t.text);
    s.e += editDistance(a, b); s.n += b.length; s.exact += a === b ? 1 : 0; s.ms += ms;
  }
}
const nAr = truth.filter((t) => t.script === 'ar').length;
console.log(`Arabic lines: ${nAr}, English lines: ${truth.length - nAr}`);
for (const [k, s] of Object.entries(stat)) console.log(`  ${k.padEnd(18)} character error ${(100 * s.e / s.n).toFixed(1)}%  lines exactly right ${s.exact}/${nAr}  ${(s.ms / nAr).toFixed(0)} ms a line`);
for (const [sh, cf] of [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.7, 0.5], [ARABIC_SHARE, ARABIC_CONF]]) {
  const isAr = (d: (typeof decide)[0]) => d.share >= sh && d.conf >= cf;
  console.log(`  decision share>=${sh} conf>=${cf}: Arabic kept ${decide.filter((d) => d.script === 'ar' && isAr(d)).length}/${nAr}, English taken for Arabic ${decide.filter((d) => d.script === 'en' && isAr(d)).length}/${truth.length - nAr}`);
}

// ---- cards: the whole path (find text, read both ways, pick, Arabic key facts)
const best = variants[process.env.WIDTH ?? 'x1.5'];
const cards: { file: string; nameAr: string; natAr: string; layout: string }[] = JSON.parse(fs.readFileSync(path.join(DATA, 'cards/truth.json'), 'utf8'));
let name = 0, nat = 0, nameC = 0, natC = 0, ms = 0;
const miss: Record<string, number> = {};
for (const c of cards) {
  let p: { data: Uint8Array; width: number; height: number } = png(path.join(DATA, 'cards', c.file));
  if (process.env.ROUGH) {
    // closer to a phone photo: half the pixels, a little blur, heavy JPEG compression
    const j = await sharp(path.join(DATA, 'cards', c.file)).resize(Math.round(p.width * +process.env.ROUGH)).blur(0.7).jpeg({ quality: 45 }).toBuffer();
    const { data, info } = await sharp(j).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    p = { data: new Uint8Array(data), width: info.width, height: info.height };
  }
  let g = rgbaToGray(p.data, p.width, p.height);
  g = preprocess(resize(g, ocrScale(g.width, g.height)), { deskew: true, binarise: false, removeLines: false }).img;
  const t0 = performance.now();
  const r = await pp.readAll(grayToRgba(g), g.width, g.height, best);
  ms += performance.now() - t0;
  const f = arabicFacts(r.arabic);
  const fn = f.find((x) => x.key === 'Name in Arabic')?.value ?? '', fa = f.find((x) => x.key === 'Nationality in Arabic')?.value ?? '';
  if (process.env.SHOW && fn !== plainArabic(c.nameAr)) console.log(c.file, c.layout, JSON.stringify(fn), JSON.stringify(plainArabic(c.nameAr)), JSON.stringify(r.arabic.map((a) => a.text)));
  if (fn === plainArabic(c.nameAr)) name++; else miss[c.layout + (fn ? ' name wrong' : ' name missing')] = (miss[c.layout + (fn ? ' name wrong' : ' name missing')] ?? 0) + 1;
  if (fa === plainArabic(c.natAr)) nat++;
  nameC += fn ? cer(fn, plainArabic(c.nameAr)) : 1; natC += fa ? cer(fa, plainArabic(c.natAr)) : 1;
}
console.log(`Cards: ${cards.length}. Name in Arabic exactly right ${name}/${cards.length} (character error ${(100 * nameC / cards.length).toFixed(1)}%), nationality ${nat}/${cards.length} (${(100 * natC / cards.length).toFixed(1)}%), ${(ms / cards.length).toFixed(0)} ms a card`);
console.log('  misses by layout:', JSON.stringify(miss));

// ---- English-only specimens: segments wrongly taken for Arabic
if (process.env.SYNTH) {
  const dir = path.resolve('../data/synth', process.env.SYNTH);
  let segs = 0, wrong = 0, pages = 0, facts = 0;
  for (const type of fs.readdirSync(dir).filter((d) => fs.statSync(path.join(dir, d)).isDirectory())) {
    for (const f of fs.readdirSync(path.join(dir, type)).filter((f) => /\.(png|jpe?g)$/.test(f)).slice(0, +(process.env.PER ?? 6))) {
      const { data, info } = await sharp(path.join(dir, type, f)).rotate().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      const p = { data: new Uint8Array(data), width: info.width, height: info.height };
      let g = rgbaToGray(p.data, p.width, p.height);
      g = preprocess(resize(g, ocrScale(g.width, g.height)), { deskew: true, binarise: false, removeLines: false }).img;
      const r = await pp.readAll(grayToRgba(g), g.width, g.height, best);
      pages++; segs += r.words.length; wrong += r.arabic.length; facts += arabicFacts(r.arabic).length;
    }
  }
  console.log(`English-only specimens: ${pages} pages, ${wrong} segments taken for Arabic (against ${segs} English words kept), Arabic facts made up: ${facts}`);
}
