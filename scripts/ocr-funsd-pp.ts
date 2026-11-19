// Run the PP-OCRv5 reader over FUNSD and cache the words (same format as ocr-funsd.ts).
// usage: npx tsx scripts/ocr-funsd-pp.ts <training|testing> <tag> <maxSide> [limit] [raw]
import fs from 'node:fs'; import path from 'node:path';
import { PNG } from 'pngjs';
import * as ort from 'onnxruntime-node';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale } from '../src/core/preprocess';
import { loadPpOcr } from '../src/core/ppocr';
import { DATA } from './funsd';

const [split = 'testing', tag = 'pp', maxSide = '1280', limitArg, raw] = process.argv.slice(2);
const root = path.join(DATA, 'funsd/dataset', split + '_data');
const outDir = path.join(DATA, 'ocr', `${split}-${tag}`);
fs.mkdirSync(outDir, { recursive: true });
const pp = await loadPpOcr(ort as any, path.join(DATA, 'ppocr/det.onnx'), path.join(DATA, 'ppocr/rec.onnx'), fs.readFileSync(path.join(DATA, 'ppocr/dict.txt'), 'utf8'), undefined, { maxSide: +maxSide, boxThresh: +(process.env.BT ?? 0.55), unclip: +(process.env.UC ?? 0.8) });
const files = fs.readdirSync(path.join(root, 'images')).filter((f) => f.endsWith('.png')).sort().slice(0, limitArg ? +limitArg : undefined);
let ms = 0;
for (const f of files) {
  const out = path.join(outDir, f.replace('.png', '.json'));
  if (fs.existsSync(out)) continue;
  const png = PNG.sync.read(fs.readFileSync(path.join(root, 'images', f)));
  let g = rgbaToGray(png.data, png.width, png.height);
  const scale = raw ? 1 : ocrScale(g.width, g.height);
  let skew = 0;
  if (!raw) ({ img: g, skew } = preprocess(resize(g, scale), { deskew: true, binarise: false, removeLines: false }));
  const t = Date.now();
  const words = (await pp.read(grayToRgba(g), g.width, g.height)).map((w) => ({ ...w, box: w.box.map((v) => v / scale) }));
  ms += Date.now() - t;
  fs.writeFileSync(out, JSON.stringify({ file: f, width: png.width, height: png.height, skew, ms: Date.now() - t, words }));
}
console.log(split, tag, files.length, 'pages', Math.round(ms / files.length), 'ms/page');
