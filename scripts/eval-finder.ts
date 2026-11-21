// How well does the outline finder (detectObject) find a card on a light surface, and does it
// leave plain scans alone? Made-up specimens are pasted, turned, onto pale surfaces with a soft
// shadow; a hit is a quad whose corners are all within 4% of the photo diagonal of the true ones.
// Then every scan in the form sets is tried: the finder should return nothing for those.
// usage: npx tsx scripts/eval-finder.ts [folder to save a few composites]
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { rgbaToGray } from '../src/core/preprocess';
import { detectCard, detectObject, detectPage, findDocument, refineQuad, type Pt, type Quad } from '../src/core/scan';

const DATA = path.resolve(import.meta.dirname, '../../data');
const out = process.argv[2];
const walk = (d: string): string[] => fs.existsSync(d) ? fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : /\.(jpe?g|png)$/i.test(e.name) ? [path.join(d, e.name)] : []) : [];
import { composite, load, err, close, SURFACES } from './composite';

const errs = { raw: [] as number[], refined: [] as number[] };

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
    if (o && close(o, truth, diag)) { errs.raw.push(err(o, truth, diag)); errs.refined.push(err(refineQuad(rgba, W, H, o as Quad), truth, diag)); }
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
    if (b && close(b, truth, diag)) { errs.raw.push(err(b, truth, diag)); errs.refined.push(err(refineQuad(rgba, W, H, b as Quad), truth, diag)); }
    if (out && total <= 6) { fs.mkdirSync(out, { recursive: true }); const svg = Buffer.from(`<svg width="${W}" height="${H}">${b ? `<polygon points="${b.map((p) => p.join(',')).join(' ')}" fill="none" stroke="#f0f" stroke-width="6"/>` : ''}</svg>`); await sharp(await sharp(buf).composite([{ input: svg }]).toBuffer()).resize(500).jpeg().toFile(path.join(out, `held-${total}.jpg`)); }
  }
  console.log(`held in a hand  outline and bright-page finders ${plain}/${total}   four-edges finder ${edges}/${total}   (${(t / total).toFixed(1)} ms a photo)`);
}
{
  const mean = (v: number[]) => (v.reduce((a, b) => a + b, 0) / v.length).toFixed(2);
  const med = (v: number[]) => [...v].sort((a, b) => a - b)[v.length >> 1].toFixed(2);
  console.log(`corner error on found cards (% of diagonal): as found mean ${mean(errs.raw)} median ${med(errs.raw)}   sharpened mean ${mean(errs.refined)} median ${med(errs.refined)}   (${errs.raw.length} cards)`);
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
