// The camera's document finder, classical against learned (DocCornerNet LEAN) and combined, on made-up
// photos with known corners (scripts/composite.ts): plain surfaces, a card held in a hand, and the
// patterned surfaces where the classical finders fail. Also photos with no card, to count outlines
// drawn where there is nothing. Prints counts only.
//   npx tsx scripts/eval-corners.ts [folder to save a few pictures]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';
import { rgbaToGray } from '../src/core/preprocess';
import { detectCard, detectObject, detectPage, refineQuad, type Quad } from '../src/core/scan';
import { CORNER_SIZE, edgeContrast, loadCornerModel, plausibleQuad } from '../src/core/corners';
import { chooseOutline, NET_CONTRAST, NET_SCORE, type Finds } from '../src/core/track';
import { composite, load, err, close, SURFACES, reseed } from './composite';

const DATA = path.resolve(import.meta.dirname, '../../data');
const out = process.argv[2];
const walk = (d: string): string[] => fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.(jpe?g|png)$/i.test(e.name) ? [path.join(d, e.name)] : []) : [];
const model = await loadCornerModel(ort as any, path.resolve('public/models/corner/doccornernet_lean.onnx'));
const MIN_SCORE = NET_SCORE;

/** a copy of the photo with `side` pixels on its long side (block average, like the browser's drawImage) */
async function copy(buf: Buffer, side: number, fill = false) {
  const s = sharp(buf).ensureAlpha();
  const r = fill ? s.resize(side, side, { fit: 'fill' }) : s.resize({ width: side, height: side, fit: 'inside' });
  const { data, info } = await r.raw().toBuffer({ resolveWithObject: true });
  return { d: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), w: info.width, h: info.height };
}
type Frac = Quad; // corners as fractions of the photo
const scaleQ = (q: Quad, w: number, h: number) => q.map(([x, y]) => [x * w, y * h]) as Quad;
const frac = (q: Quad, w: number, h: number) => q.map(([x, y]) => [x / w, y / h]) as Quad;


type Parts = { obj: Frac | null; edge: Frac | null; page: Frac | null; l: Frac; score: number; plausible: boolean; contrast: number };
/** every finder's answer for one photo (classical ones on a 320 px copy, the network on 224 x 224) */
async function parts(buf: Buffer): Promise<Parts> {
  const s = await copy(buf, 320);
  const f = (q: Quad | null) => (q ? frac(q, s.w, s.h) : null);
  const obj = f(detectObject(s.d, s.w, s.h));
  const edge = obj ? null : f(detectCard(s.d, s.w, s.h));
  const page = obj ? null : f(detectPage({ width: s.w, height: s.h, data: rgbaToGray(s.d, s.w, s.h).data }));
  const m = await copy(buf, CORNER_SIZE, true);
  const r = await model.find(m.d);
  const contrast = edgeContrast(s.d, s.w, s.h, scaleQ(r.quad, s.w, s.h));
  contrasts.push({ c: contrast, ok: r.score >= T && plausibleQuad(r.quad) });
  return { obj, edge, page, l: r.quad, score: r.score, plausible: plausibleQuad(r.quad), contrast };
}
async function sharpen(buf: Buffer, q: Frac | null): Promise<Frac | null> {
  if (!q) return null;
  const b = await copy(buf, 960);
  return frac(refineQuad(b.d, b.w, b.h, scaleQ(q, b.w, b.h)), b.w, b.h);
}
const T = MIN_SCORE;
let contrasts: { c: number; ok: boolean; hit?: boolean }[] = [];
const CMIN = NET_CONTRAST;
const learnedOk = (p: Parts) => (p.score >= T && p.plausible && p.contrast >= CMIN ? p.l : null);
const near = (a: Quad, b: Quad, d: number) => Math.max(...a.map((p) => Math.min(...b.map((c) => Math.hypot(p[0] - c[0], p[1] - c[1]))))) < d;
const finds = (p: Parts): Finds => ({ outline: p.obj, edges: p.edge, page: p.page, net: { quad: p.l, score: p.score, plausible: p.plausible, contrast: p.contrast } });
const rules: Record<string, (p: Parts) => Frac | null> = {
  'classical (today)': (p) => p.obj ?? p.edge ?? p.page,
  'learned alone': learnedOk,
  'shipped: chooseOutline': (p) => chooseOutline(finds(p))?.q ?? null,
  'shipped, strong only (auto-capture)': (p) => { const c = chooseOutline(finds(p)); return c?.strong ? c.q : null; },
};
const ways = Object.keys(rules);
type Way = string;
async function all(buf: Buffer): Promise<Record<Way, Frac | null>> {
  const p = await parts(buf);
  const out: Record<Way, Frac | null> = {};
  for (const w of ways) out[w] = await sharpen(buf, rules[w](p));
  return out;
}

const cards = walk(path.join(DATA, 'synth/train')).filter((f) => /flat/.test(f)).slice(0, 60);
const conditions: [string, (c: string) => ReturnType<typeof composite>][] = [
  ...Object.entries(SURFACES).map(([n, bg]) => [n, (c: string) => composite(c, bg)] as [string, any]),
  ['held in a hand', (c) => composite(c, [150, 150, 150], 1200, 1600, true)],
  ['stripes', (c) => composite(c, [235, 230, 220], 1200, 1600, false, 'stripes')],
  ['grid', (c) => composite(c, [240, 240, 236], 1200, 1600, false, 'grid')],
  ['wood grain', (c) => composite(c, [150, 105, 60], 1200, 1600, false, 'wood grain')],
];
const hits: Record<Way, number> = Object.fromEntries(ways.map((w) => [w, 0])) as any;
const errsOf: Record<Way, number[]> = Object.fromEntries(ways.map((w) => [w, []])) as any;
let total = 0, saved = 0;
const table: string[] = [];
reseed(11);
for (const [name, make] of conditions) {
  const h: Record<Way, number> = Object.fromEntries(ways.map((w) => [w, 0])) as any;
  let n = 0;
  for (const c of cards.filter((_, i) => i % 3 === 0)) {
    const { buf, truth, W, H } = await make(c);
    const diag = Math.hypot(W, H);
    const r = await all(buf);
    n++; total++;
    for (const w of ways) {
      const q = r[w] && scaleQ(r[w]!, W, H);
      if (q && close(q, truth, diag)) { h[w]++; hits[w]++; errsOf[w].push(err(q, truth, diag)); }
    }
    if (out && saved < 24 && n <= 2) {
      fs.mkdirSync(out, { recursive: true });
      const poly = (q: Quad | null, col: string) => (q ? `<polygon points="${scaleQ(q, W, H).map((p) => p.join(',')).join(' ')}" fill="none" stroke="${col}" stroke-width="6"/>` : '');
      const svg = Buffer.from(`<svg width="${W}" height="${H}">${poly(r['classical (today)'], '#f0f')}${poly(r['learned'], '#0c0')}</svg>`);
      await sharp(await sharp(buf).composite([{ input: svg }]).toBuffer()).resize(400).jpeg().toFile(path.join(out, `${name.replace(/ /g, '-')}-${n}.jpg`));
      saved++;
    }
  }
  table.push(`${name.padEnd(15)} ` + ways.map((w) => `${h[w]}/${n}`.padStart(6)).join(' '));
}
const q = (v: number[], f: number) => v.length ? v.sort((a, b) => a - b)[Math.floor(f * (v.length - 1))].toFixed(2) : '-';
console.log(`outline contrast of the network's answers on photos with a card: 10th percentile ${q(contrasts.filter((c) => c.ok).map((c) => c.c), 0.1)}, median ${q(contrasts.filter((c) => c.ok).map((c) => c.c), 0.5)}`);
console.log(`${''.padEnd(15)} ` + ways.map((_, i) => `   (${i + 1})`).join(' '));
for (const l of table) console.log(l);
console.log(`${'all'.padEnd(15)} ` + ways.map((w) => `${hits[w]}/${total}`.padStart(6)).join(' '));
ways.forEach((w, i) => { const v = errsOf[w].sort((a, b) => a - b); console.log(`(${i + 1}) ${w}: corner error on found cards median ${v.length ? v[v.length >> 1].toFixed(2) : '-'}% of the diagonal`); });

// photos with no card at all: an outline here is a false alarm
{
  const fa: Record<Way, number> = Object.fromEntries(ways.map((w) => [w, 0])) as any;
  let n = 0;
  for (const [name, bg, pat, held] of [['white', [250, 250, 248]], ['dark wood', [92, 64, 44]], ['stripes', [235, 230, 220], 'stripes'], ['grid', [240, 240, 236], 'grid'], ['wood grain', [150, 105, 60], 'wood grain'], ['busy', [150, 150, 150], undefined, true]] as const) {
    for (let i = 0; i < 8; i++) {
      const { buf } = await composite(cards[i], bg as any, 1200, 1600, !!held, pat as any, false);
      const r = await all(buf);
      n++;
      for (const w of ways) if (r[w]) fa[w]++;
      void name;
    }
  }
  console.log(`no card in the photo (${n} photos), outline drawn anyway: ` + ways.map((w, i) => `(${i + 1}) ${fa[w]}`).join('  '));
}
// speed of the network on this computer
{
  const s = await copy(await sharp({ create: { width: 1200, height: 1600, channels: 3, background: '#888' } }).jpeg().toBuffer(), CORNER_SIZE, true);
  const t = performance.now();
  for (let i = 0; i < 20; i++) await model.find(s.d);
  console.log(`network: ${((performance.now() - t) / 20).toFixed(1)} ms a frame (Node, CPU)`);
}

// flat scans (forms, specimens lying flat in the picture): the network should not propose a cut inside them
{
  for (const [name, dir] of [['FUNSD test scans', 'funsd/dataset/testing_data/images'], ['modern forms', 'forms20/blank-img'], ['flat specimens', 'synth/train']] as const) {
    const files = walk(path.join(DATA, dir)).filter((f) => !/synth/.test(f) || /flat/.test(f)).slice(0, 60);
    let cut = 0, scores: number[] = [], gaps: number[] = [], cts: number[] = [];
    for (const f of files) {
      const m = await copy(fs.readFileSync(f), CORNER_SIZE, true);
      const r = await model.find(m.d);
      scores.push(r.score);
      const frame: Quad = [[0, 0], [1, 0], [1, 1], [0, 1]];
      const gap = Math.max(...r.quad.map((p, i) => Math.hypot(p[0] - frame[i][0], p[1] - frame[i][1])));
      gaps.push(gap);
      const sm = await copy(fs.readFileSync(f), 320);
      const ct = edgeContrast(sm.d, sm.w, sm.h, scaleQ(r.quad, sm.w, sm.h));
      if (r.score >= MIN_SCORE && plausibleQuad(r.quad) && ct >= CMIN) cut++;
      cts.push(ct);
    }
    scores.sort((a, b) => a - b);
    console.log(`${name.padEnd(18)} ${files.length} files: the network proposes a cut in ${cut} (median score ${scores[scores.length >> 1].toFixed(2)}); outline contrast median ${q(cts, 0.5)}, 90th percentile ${q(cts, 0.9)}`);
  }
}
