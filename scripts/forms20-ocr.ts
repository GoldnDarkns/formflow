// Read page images with either reader, the same way the app does, with a cache in ../data/forms20.
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import * as ortNode from 'onnxruntime-node';
import { createWorker, PSM } from 'tesseract.js';
import type { Box, OcrWord } from '../src/core/types';
import { rgbaToGray, grayToRgba, preprocess, resize, ocrScale } from '../src/core/preprocess';
import { wordsFromTesseract, cleanWords } from '../src/core/ocr';
import { loadPpOcr } from '../src/core/ppocr';
import { DATA } from './funsd';

export const ROOT = path.resolve(import.meta.dirname, '..');
export type Reader = 'tesseract' | 'ppocr';
export interface Cached { width: number; height: number; ms: number; words: OcrWord[] }

async function gray(file: string) {
  const { data, info } = await sharp(file).greyscale().raw().toBuffer({ resolveWithObject: true });
  const rgba = new Uint8Array(info.width * info.height * 4);
  for (let i = 0; i < data.length; i++) { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = data[i]; rgba[i * 4 + 3] = 255; }
  return rgbaToGray(rgba, info.width, info.height);
}

/** OCR `files` (names inside `dir`) with `reader`; results cached under ../data/forms20/<cacheName>-<reader>. */
export async function ocrAll(reader: Reader, dir: string, files: string[], cacheName = 'ocr'): Promise<Map<string, Cached>> {
  const cdir = path.join(DATA, 'forms20', `${cacheName}-${reader}`);
  fs.mkdirSync(cdir, { recursive: true });
  const out = new Map<string, Cached>();
  let worker: Awaited<ReturnType<typeof createWorker>> | null = null;
  let pp: Awaited<ReturnType<typeof loadPpOcr>> | null = null;
  for (const file of files) {
    const cache = path.join(cdir, file.replace(/\.\w+$/, '.json'));
    if (fs.existsSync(cache)) { const j = JSON.parse(fs.readFileSync(cache, 'utf8')); out.set(file, { ...j, words: cleanWords(j.words) }); continue; }
    let g = await gray(path.join(dir, file));
    const W = g.width, H = g.height;
    const scale = ocrScale(W, H);
    let words: OcrWord[];
    const t = Date.now();
    if (reader === 'tesseract') {
      // same steps as the app: resize, deskew, binarise, remove ruling lines, sparse-text Tesseract
      ({ img: g } = preprocess(resize(g, scale)));
      worker ??= await createWorker('eng', 1, { langPath: path.join(ROOT, 'node_modules/@tesseract.js-data/eng/4.0.0_best_int'), cacheMethod: 'none' });
      await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
      const enc = new PNG({ width: g.width, height: g.height });
      enc.data = Buffer.from(grayToRgba(g));
      const { data } = await worker.recognize(PNG.sync.write(enc), {}, { blocks: true });
      words = wordsFromTesseract(data.blocks as any, scale);
    } else {
      ({ img: g } = preprocess(resize(g, scale), { deskew: true, binarise: false, removeLines: false }));
      pp ??= await loadPpOcr(ortNode as any, path.join(ROOT, 'public/models/ppocr/det.onnx'), path.join(ROOT, 'public/models/ppocr/rec.onnx'), fs.readFileSync(path.join(ROOT, 'public/models/ppocr/dict.txt'), 'utf8'), undefined, { maxSide: 1600, boxThresh: 0.55, unclip: 0.8 });
      words = (await pp.read(grayToRgba(g), g.width, g.height)).map((w) => ({ ...w, box: w.box.map((v) => v / scale) as Box }));
    }
    const c = { file, width: W, height: H, ms: Date.now() - t, words };
    fs.writeFileSync(cache, JSON.stringify(c));
    out.set(file, { ...c, words: cleanWords(words) });
    process.stdout.write(`${reader} ${file} ${words.length} words ${c.ms} ms\n`);
  }
  await worker?.terminate();
  return out;
}
