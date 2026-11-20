// How well does the outline finder (detectObject) find a card on a light surface, and does it
// leave plain scans alone? Made-up specimens are pasted, turned, onto pale surfaces with a soft
// shadow; a hit is a quad whose corners are all within 4% of the photo diagonal of the true ones.
// Then every scan in the form sets is tried: the finder should return nothing for those.
// usage: npx tsx scripts/eval-finder.ts [folder to save a few composites]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { rgbaToGray } from '../src/core/preprocess';
import { detectCard, detectObject, detectPage, findDocument, type Pt } from '../src/core/scan';

const DATA = path.resolve(import.meta.dirname, '../../data');
const out = process.argv[2];
const walk = (d: string): string[] => fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.(jpe?g|png)$/i.test(e.name) ? [path.join(d, e.name)] : []) : [];
let seed = 7;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

const SURFACES: Record<string, [number, number, number]> = { white: [250, 250, 248], cream: [236, 230, 214], grey: [196, 198, 202], 'dark wood': [92, 64, 44], black: [28, 28, 30] };

async function composite(card: string, bg: [number, number, number], W = 1200, H = 1600, held = false) {
  const ang = (rnd() - 0.5) * 30;
  const cw = Math.round(W * (0.55 + rnd() * 0.25));
  const img = await sharp(card).resize(cw).toBuffer({ resolveWithObject: true });
  const ch = img.info.height;
  const cx = W / 2 + (rnd() - 0.5) * W * 0.1, cy = H / 2 + (rnd() - 0.5) * H * 0.3;
  const a = (ang * Math.PI) / 180;
  const truth = ([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]] as Pt[]).map(([x, y]) => [cx + x * Math.cos(a) - y * Math.sin(a), cy + x * Math.sin(a) + y * Math.cos(a)] as Pt);
  const rot = await sharp(img.data).ensureAlpha().rotate(ang, { background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toBuffer({ resolveWithObject: true });
  // a soft shadow under the card, and light falling off towards one side
  const shadow = await sharp(rot.data).ensureAlpha().extractChannel('alpha').blur(6).linear(0.25).toBuffer();
  const shade = await sharp({ create: { width: rot.info.width, height: rot.info.height, channels: 3, background: '#000' } }).joinChannel(shadow).png().toBuffer();
  const grad = Buffer.from(`<svg width="${W}" height="${H}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.12"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#g)"/></svg>`);
  const left = Math.round(cx - rot.info.width / 2), top = Math.round(cy - rot.info.height / 2);
  // "held in a hand": a busy background of coloured shapes behind the card, and three finger-like blobs over one of its edges
  const shapes = held ? Array.from({ length: 14 }, () => { const x = rnd() * W, y = rnd() * H, r = 60 + rnd() * 260, c = `rgb(${(40 + rnd() * 200) | 0},${(40 + rnd() * 200) | 0},${(40 + rnd() * 200) | 0})`; return rnd() < 0.5 ? `<ellipse cx="${x}" cy="${y}" rx="${r}" ry="${r * (0.4 + rnd())}" fill="${c}"/>` : `<rect x="${x - r}" y="${y - r / 2}" width="${r * 2}" height="${r * (0.5 + rnd())}" fill="${c}" transform="rotate(${rnd() * 90} ${x} ${y})"/>`; }).join('') : '';
  const clutter = Buffer.from(`<svg width="${W}" height="${H}">${shapes}</svg>`);
  const side = truth[1], next = truth[2]; // fingers over the right-hand edge
  const fingers = held ? [0.25, 0.5, 0.75].map((f) => { const x = side[0] + (next[0] - side[0]) * f, y = side[1] + (next[1] - side[1]) * f; return `<ellipse cx="${x + 30}" cy="${y}" rx="95" ry="38" fill="rgb(214,168,140)" transform="rotate(${ang + (rnd() - 0.5) * 20} ${x} ${y})"/>`; }).join('') : '';
  const hand = Buffer.from(`<svg width="${W}" height="${H}">${fingers}</svg>`);
  const noise = Buffer.alloc(W * H * 3);
  for (let i = 0; i < noise.length; i++) noise[i] = 128 + (rnd() - 0.5) * 12;
  const buf = await sharp({ create: { width: W, height: H, channels: 3, background: { r: bg[0], g: bg[1], b: bg[2] } } })
    .composite([{ input: clutter }, { input: shade, left: left + 5, top: top + 7 }, { input: rot.data, left, top }, { input: hand }, { input: grad }, { input: noise, raw: { width: W, height: H, channels: 3 }, blend: 'soft-light' }])
    .jpeg({ quality: 85 }).toBuffer();
  return { buf, truth, W, H };
}

const load = async (src: Buffer | string) => { const { data, info } = await sharp(src).rotate().flatten({ background: '#fff' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true }); return { rgba: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), W: info.width, H: info.height }; };
const close = (q: Pt[], t: Pt[], diag: number) => {
  // corners may come back in a different starting order when the card is turned: match each to its nearest
  return t.every((p) => Math.min(...q.map((c) => Math.hypot(c[0] - p[0], c[1] - p[1]))) < diag * 0.04);
};

const cards = walk(path.join(DATA, 'synth/train')).filter((f) => /flat/.test(f)).slice(0, 60);
console.log(`${cards.length} flat specimens`);
let saved = 0, ms = 0, runs = 0;
for (const [name, bg] of Object.entries(SURFACES)) {
  let hitObj = 0, hitBoth = 0, hitOld = 0, total = 0;
  for (const c of cards.filter((_, i) => i % 3 === 0)) {
    const { buf, truth } = await composite(c, bg);
    const { rgba, W, H } = await load(buf);
    const diag = Math.hypot(W, H), g = rgbaToGray(rgba, W, H);
    const t = performance.now();
    const o = detectObject(rgba, W, H);
    ms += performance.now() - t; runs++;
    const old = detectPage(g), both = findDocument(rgba, W, H, g);
    total++;
    if (o && close(o, truth, diag)) hitObj++;
    if (old && close(old, truth, diag)) hitOld++;
    if (both && close(both, truth, diag)) hitBoth++;
    if (out && saved < 10 && total <= 2) { fs.mkdirSync(out, { recursive: true }); const q = both; const svg = Buffer.from(`<svg width="${W}" height="${H}">${q ? `<polygon points="${q.map((p) => p.join(',')).join(' ')}" fill="none" stroke="#f0f" stroke-width="6"/>` : ''}</svg>`); await sharp(await sharp(buf).composite([{ input: svg }]).toBuffer()).resize(500).jpeg().toFile(path.join(out, `${name.replace(' ', '-')}-${total}.jpg`)); saved++; }
  }
  console.log(`${name.padEnd(10)} bright-page finder ${hitOld}/${total}   outline finder ${hitObj}/${total}   both together ${hitBoth}/${total}`);
}
console.log(`outline finder: ${(ms / runs).toFixed(1)} ms a photo`);

// a card held up in front of a busy background, fingers over one edge
{
  let plain = 0, edges = 0, total = 0, t = 0;
  for (const c of cards.filter((_, i) => i % 2 === 0)) {
    const { buf, truth } = await composite(c, [150, 150, 150], 1200, 1600, true);
    const { rgba, W, H } = await load(buf);
    const diag = Math.hypot(W, H);
    const a = findDocument(rgba, W, H, rgbaToGray(rgba, W, H));
    const t0 = performance.now();
    const b = detectCard(rgba, W, H);
    t += performance.now() - t0;
    total++;
    if (a && close(a, truth, diag)) plain++;
    if (b && close(b, truth, diag)) edges++;
    if (out && total <= 6) { fs.mkdirSync(out, { recursive: true }); const svg = Buffer.from(`<svg width="${W}" height="${H}">${b ? `<polygon points="${b.map((p) => p.join(',')).join(' ')}" fill="none" stroke="#f0f" stroke-width="6"/>` : ''}</svg>`); await sharp(await sharp(buf).composite([{ input: svg }]).toBuffer()).resize(500).jpeg().toFile(path.join(out, `held-${total}.jpg`)); }
  }
  console.log(`held in a hand  outline and bright-page finders ${plain}/${total}   four-edges finder ${edges}/${total}   (${(t / total).toFixed(1)} ms a photo)`);
}

// scans: nothing should be cut
for (const [name, dir] of [['FUNSD test scans', 'funsd/dataset/testing_data/images'], ['modern forms', 'forms20/blank-img'], ['flat specimens', 'synth/train']] as const) {
  const files = walk(path.join(DATA, dir)).filter((f) => !/synth/.test(f) || /flat/.test(f)).slice(0, 80);
  let cutOld = 0, cutNew = 0;
  for (const f of files) {
    const { rgba, W, H } = await load(f);
    if (detectPage(rgbaToGray(rgba, W, H))) cutOld++;
    else if (detectObject(rgba, W, H)) cutNew++;
  }
  console.log(`${name.padEnd(18)} ${files.length} files: cut by the bright-page finder ${cutOld}, cut by the outline finder ${cutNew}`);
}
