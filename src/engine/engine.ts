// Main-thread orchestrator: image -> pre-processing (worker) -> Tesseract.js ->
// layout classifiers (ONNX Runtime Web) -> MRZ check -> optional Florence-2.
import * as ort from 'onnxruntime-web/webgpu';
import { createWorker, PSM, type Worker as TWorker } from 'tesseract.js';
import type { Analysis } from '../core/pipeline';
import { analyse, type Classifiers } from '../core/pipeline';
import { loadClassifiers } from '../core/classifier';
import { wordsFromTesseract, cleanWords } from '../core/ocr';
import { parseMrz, findMrzLines, type MrzResult } from '../core/mrz';
import type { OcrWord, SecondReading } from '../core/types';
import type { PreprocessOptions } from '../core/preprocess';
import { classifyDocument, DOC_LABEL, ID_TYPES, type Classification, type DocType } from '../core/classify';
import { extractFacts, type Fact } from '../core/facts';
import type { Quad } from '../core/scan';
import type { VisionOpinion } from '../core/critic-vision';
import type { PreRequest, PreResponse } from './preprocess.worker';

import { matchTemplate, readTemplate, type Template, type TemplateMatch } from '../core/template';
export const BASE = new URL(import.meta.env.BASE_URL, location.href).href;
ort.env.wasm.wasmPaths = BASE + 'ort/';
ort.env.wasm.numThreads = 1;


export interface Step {
  name: string;
  state: 'pending' | 'running' | 'done' | 'skipped' | 'error';
  ms?: number;
  detail?: string;
}

export interface DocResult {
  template?: TemplateMatch; // set when the page was read with a known-form template
  kind: 'form' | 'id';
  docType: DocType;
  classification: Classification;
  facts: Fact[];
  original: string; // the photo as taken (small), for the before/after view
  quad: Quad | null; // page corners in the photo, 0..1
  lit: boolean;
  id: string;
  fileName: string;
  width: number;
  height: number;
  view: string; // data URL of the deskewed colour image (overlay coordinates)
  processedView: string; // data URL of the cleaned image the OCR saw
  skew: number;
  words: OcrWord[];
  reader: 'tesseract' | 'ppocr'; // which reader produced `words` (picks the calibration curve)
  analysis: Analysis;
  mrz: MrzResult | null;
  second?: SecondReading[];
  secondMs?: number;
  steps: Step[];
  totalMs: number;
  ocrScale: number;
  grayRgba: ArrayBuffer;
  grayW: number;
  grayH: number;
}

const MAX_SIDE = 2400;

let tess: Promise<TWorker> | null = null;
let tessMrz: Promise<TWorker> | null = null;
let clf: Promise<Classifiers> | null = null;
let pre: Worker | null = null;

function tesseract(): Promise<TWorker> {
  return (tess ??= (async () => {
    const w = await createWorker('eng', 1, { workerPath: BASE + 'tesseract/worker.min.js', corePath: BASE + 'tesseract/core', langPath: BASE + 'tesseract/lang', workerBlobURL: false });
    await w.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: '1' });
    return w;
  })());
}
function tesseractMrz(): Promise<TWorker> {
  return (tessMrz ??= (async () => {
    const w = await createWorker('eng', 1, { workerPath: BASE + 'tesseract/worker.min.js', corePath: BASE + 'tesseract/core', langPath: BASE + 'tesseract/lang', workerBlobURL: false }, { load_system_dawg: '0', load_freq_dawg: '0' } as any);
    await w.setParameters({ tessedit_pageseg_mode: PSM.SINGLE_BLOCK, tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<' });
    return w;
  })());
}
// ---------------- templates for known forms ----------------
const USER_TEMPLATES = 'ff.templates';
let library: Promise<Template[]> | null = null;
/** Templates the person taught on this device (kept in the browser's local storage, never uploaded). */
export function userTemplates(): Template[] {
  try { return JSON.parse(localStorage.getItem(USER_TEMPLATES) ?? '[]'); } catch { return []; }
}
export function saveUserTemplate(t: Template) {
  localStorage.setItem(USER_TEMPLATES, JSON.stringify([t, ...userTemplates().filter((x) => x.id !== t.id)]));
}
export function deleteUserTemplate(id: string) {
  localStorage.setItem(USER_TEMPLATES, JSON.stringify(userTemplates().filter((x) => x.id !== id)));
}
/** The person's own templates first, then the built-in library of public forms. */
export function templates(): Promise<Template[]> {
  library ??= fetch(BASE + 'templates/library.json').then((r) => r.json()).then((j) => j.templates as Template[]).catch(() => []);
  return library.then((l) => [...userTemplates(), ...l]);
}
const templateAnalysis = (m: TemplateMatch, doc: { words: OcrWord[]; width: number; height: number }): Analysis => readTemplate(m, doc.words, doc.width, doc.height);
const templateClass = (cls: Classification, m: TemplateMatch): Classification => {
  const ev = { type: 'form' as DocType, weight: 9, why: `Matches the known form "${m.template.name}" (${Math.round(m.score * 100)}% of its printed words are in place)` };
  return { ...cls, type: 'form', confidence: Math.max(cls.type === 'form' ? cls.confidence : 0, 0.99), evidence: [ev, ...cls.evidence.filter((e) => e.type === 'form')], all: [ev, ...cls.all] };
};
/** Read an already-processed page with one template (used right after teaching it). Null when it does not fit. */
export function applyTemplate(doc: DocResult, t: Template): DocResult | null {
  const m = matchTemplate([t], doc.words, doc.width, doc.height);
  if (!m) return null;
  const analysis = templateAnalysis(m, doc);
  return { ...doc, kind: 'form', docType: 'form', template: m, analysis, classification: templateClass(doc.classification, m), facts: extractFacts('form', doc.words, analysis.fields, doc.mrz) };
}

export function classifiers(): Promise<Classifiers> {
  return (clf ??= loadClassifiers(ort as any, BASE + 'models/entity_gbdt.onnx', BASE + 'models/link_gbdt.onnx', { executionProviders: ['wasm'] }));
}
/** warm up engines in the background so the first scan is fast */
export function warmUp() {
  tesseract().catch(() => (tess = null));
  classifiers().catch(() => (clf = null));
}

function runPre(req: PreRequest): Promise<PreResponse> {
  pre ??= new Worker(new URL('./preprocess.worker.ts', import.meta.url), { type: 'module' });
  return new Promise((res, rej) => {
    pre!.onmessage = (e) => res(e.data);
    pre!.onerror = (e) => rej(new Error(e.message));
    pre!.postMessage(req, [req.rgba]);
  });
}

async function loadBitmap(file: Blob): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' } as any);
  } catch {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    await img.decode();
    URL.revokeObjectURL(url);
    return createImageBitmap(img);
  }
}

function canvas(w: number, h: number) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}
function rgbaToCanvas(buf: ArrayBuffer, w: number, h: number) {
  const c = canvas(w, h);
  c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(buf), w, h), 0, 0);
  return c;
}

/** Locate the MRZ band from first-pass OCR lines (rows dense with '<'). */
function mrzBand(words: OcrWord[], scale: number, H: number): [number, number] | null {
  const lines = new Map<number, OcrWord[]>();
  for (const w of words) {
    if (!lines.has(w.line)) lines.set(w.line, []);
    lines.get(w.line)!.push(w);
  }
  const ys: number[] = [];
  for (const ws of lines.values()) {
    const t = ws.map((w) => w.text).join('');
    if ((t.match(/</g)?.length ?? 0) >= 3 || /[A-Z0-9]{2,}<{2,}/.test(t)) ys.push(...ws.flatMap((w) => [w.box[1], w.box[3]]));
  }
  if (!ys.length) return null;
  const y0 = Math.min(...ys) * scale, y1 = Math.max(...ys) * scale;
  const pad = (y1 - y0) * 0.6 + 10;
  return [Math.max(0, y0 - pad), Math.min(H, y1 + pad)];
}

const mrzScore = (m: MrzResult | null) => (m ? m.checks.filter((c) => c.ok).length + (m.valid ? 10 : 0) : -1);

/**
 * Read the MRZ from several crops (tight band, a taller band that catches an ID
 * card's filler-free first line, and the binarised image) and keep the reading
 * whose check digits agree best. Stops as soon as every check digit passes.
 */
async function readMrz(sources: HTMLCanvasElement[], band: [number, number]): Promise<MrzResult | null> {
  const [y0, y1] = band;
  const lineH = (y1 - y0) / 3;
  const bands: [number, number][] = [[y0, y1], [Math.max(0, y0 - lineH * 1.5), y1]];
  const w = await tesseractMrz();
  let best: MrzResult | null = null;
  for (const src of sources)
    for (const [a, b] of bands) {
      const c = canvas(src.width, Math.round(b - a));
      c.getContext('2d')!.drawImage(src, 0, a, src.width, b - a, 0, 0, src.width, b - a);
      const { data } = await w.recognize(c);
      const lines = findMrzLines(data.text);
      const m = lines.length >= 2 ? parseMrz(lines) : null;
      if (mrzScore(m) > mrzScore(best)) best = m;
      if (best?.valid) return best;
    }
  return best;
}

export interface Live { original?: string; quad?: Quad | null; flat?: string }
function shrink(c: HTMLCanvasElement, max: number) {
  const k = Math.min(1, max / Math.max(c.width, c.height));
  const o = canvas(Math.round(c.width * k), Math.round(c.height * k));
  o.getContext('2d')!.drawImage(c, 0, 0, o.width, o.height);
  return o;
}

export async function processDocument(file: Blob, fileName: string, opts: PreprocessOptions, onStep: (s: Step[]) => void, onLive?: (l: Live) => void): Promise<DocResult> {
  const t0 = performance.now();
  const steps: Step[] = [
    { name: 'Open image', state: 'running' },
    { name: 'Clean up the scan', state: 'pending' },
    { name: 'Read the text', state: 'pending' },
    { name: 'Recognise the document', state: 'pending' },
    { name: 'Pull out the fields', state: 'pending' },
  ];
  const upd = (i: number, s: Partial<Step>) => {
    Object.assign(steps[i], s);
    onStep(steps.map((x) => ({ ...x })));
  };
  let t = performance.now();
  const bmp = await loadBitmap(file);
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const W0 = Math.round(bmp.width * k), H0 = Math.round(bmp.height * k);
  const src = canvas(W0, H0);
  const sctx = src.getContext('2d', { willReadFrequently: true })!;
  sctx.fillStyle = '#fff';
  sctx.fillRect(0, 0, W0, H0);
  sctx.drawImage(bmp, 0, 0, W0, H0);
  const original = shrink(src, 900).toDataURL('image/jpeg', 0.7);
  onLive?.({ original });
  const rgba = sctx.getImageData(0, 0, W0, H0).data.buffer as ArrayBuffer;
  upd(0, { state: 'done', ms: performance.now() - t, detail: `${W0}×${H0}` });

  upd(1, { state: 'running' });
  const tessP = tesseract();
  const clfP = classifiers();
  const p = await runPre({ rgba, width: W0, height: H0, opts });
  const W = p.baseW, H = p.baseH;
  const quad = p.quad ? (p.quad.map(([x, y]) => [x / W0, y / H0]) as Quad) : null;

  // colour view of the flattened page, rotated by the skew angle so overlays line up
  const flat = rgbaToCanvas(p.color, W, H);
  const view = canvas(W, H);
  const vctx = view.getContext('2d')!;
  vctx.fillStyle = '#fff';
  vctx.fillRect(0, 0, W, H);
  vctx.translate(W / 2, H / 2);
  vctx.rotate((-p.skew * Math.PI) / 180);
  vctx.drawImage(flat, -W / 2, -H / 2);
  const viewUrl = view.toDataURL('image/jpeg', 0.9);
  onLive?.({ quad, flat: viewUrl });
  upd(1, { state: 'done', ms: p.ms, detail: [quad ? 'page cut out' : '', p.lit ? 'lighting fixed' : '', p.skew ? `straightened ${Math.abs(p.skew).toFixed(1)}°` : ''].filter(Boolean).join(', ') || 'already clean' });
  // give the cut-out animation a moment on screen when a page was found
  if (quad) await new Promise((r) => setTimeout(r, 900));
  const ocrCanvas = rgbaToCanvas(p.ocr, p.width, p.height);

  t = performance.now();
  upd(2, { state: 'running' });
  let words: OcrWord[];
  let readBy = 'Tesseract';
  if (opts.reader === 'ppocr') {
    // neural reader works on the deskewed grayscale page (it does not want a binarised image)
    const r = await ppRead(p.gray.slice(0), p.width, p.height);
    words = cleanWords(r.words.map((w) => ({ ...w, box: w.box.map((v) => v / p.scale) as OcrWord['box'] })));
    readBy = `PP-OCRv5 on ${r.device === 'webgpu' ? 'GPU' : 'CPU'}`;
  } else {
    const tw = await tessP;
    const { data } = await tw.recognize(ocrCanvas, {}, { blocks: true });
    words = cleanWords(wordsFromTesseract(data.blocks as any, p.scale));
  }
  upd(2, { state: 'done', ms: performance.now() - t, detail: `${words.length} words, ${readBy}` });

  // machine-readable zone first: it is the strongest clue to the document type
  t = performance.now();
  upd(3, { state: 'running' });
  let mrz: MrzResult | null = null;
  const band = mrzBand(words, p.scale, p.height);
  if (band) mrz = await readMrz([rgbaToCanvas(p.gray, p.width, p.height), ocrCanvas], band);
  let cls = classifyDocument(words, W, H, mrz);
  // a page whose printed words line up with a known form is that form, whatever the sorter thought
  const tm = mrz?.valid ? null : matchTemplate(await templates(), words, W, H);
  if (tm) cls = templateClass(cls, tm);
  upd(3, { state: 'done', ms: performance.now() - t, detail: `${DOC_LABEL[cls.type]}, ${Math.round(cls.confidence * 100)}% sure` });

  t = performance.now();
  upd(4, { state: 'running' });
  const kind = ID_TYPES.includes(cls.type) ? 'id' : 'form';
  const c = await clfP;
  const analysis = tm ? templateAnalysis(tm, { words, width: W, height: H }) : await analyse(words, W, H, c, undefined, kind, opts.reader === 'ppocr' ? 'ppocr' : 'tesseract');
  const facts = extractFacts(cls.type, words, analysis.fields, mrz);
  upd(4, { state: 'done', ms: performance.now() - t, detail: tm ? `${analysis.fields.filter((f) => f.value).length} of ${analysis.fields.length} boxes filled, template: ${tm.template.name}` : `${analysis.fields.filter((f) => f.value).length} fields` });

  return {
    template: tm ?? undefined,
    kind,
    docType: cls.type,
    classification: cls,
    facts,
    id: crypto.randomUUID(),
    fileName,
    width: W,
    height: H,
    view: viewUrl,
    original,
    quad,
    lit: p.lit,
    processedView: ocrCanvas.toDataURL('image/png'),
    skew: p.skew,
    words,
    reader: opts.reader === 'ppocr' ? 'ppocr' : 'tesseract',
    analysis,
    mrz,
    steps,
    totalMs: performance.now() - t0 - (quad ? 900 : 0),
    ocrScale: p.scale,
    grayRgba: p.gray,
    grayW: p.width,
    grayH: p.height,
  };
}

/** The clerk corrected the document type: redo layout analysis and key facts for it. */
export async function retype(doc: DocResult, type: DocType, second?: SecondReading[]): Promise<DocResult> {
  const kind = ID_TYPES.includes(type) ? 'id' : 'form';
  const analysis = await analyse(doc.words, doc.width, doc.height, await classifiers(), second, kind, doc.reader);
  return { ...doc, template: undefined, kind, docType: type, analysis, facts: extractFacts(type, doc.words, analysis.fields, doc.mrz), classification: { ...doc.classification, type, confidence: 1, evidence: [{ type, weight: 9, why: 'Set by you' }], all: [{ type, weight: 9, why: 'Set by you' }] } };
}

// ---------------- PP-OCRv5 reader ----------------
let ow: Worker | null = null;
let oReady: Promise<string> | null = null;
let oId = 0;
const oPending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
export function loadPpReader(): Promise<string> {
  if (oReady) return oReady;
  ow = new Worker(new URL('./ocr.worker.ts', import.meta.url), { type: 'module' });
  oReady = new Promise((res, rej) => {
    ow!.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') res(m.device);
      else if (m.type === 'result') oPending.get(m.id)?.res(m), oPending.delete(m.id);
      else if (m.type === 'error') {
        if (m.id !== undefined && oPending.has(m.id)) oPending.get(m.id)!.rej(new Error(m.error)), oPending.delete(m.id);
        else { oReady = null; rej(new Error(m.error)); }
      }
    };
  });
  ow.postMessage({ type: 'load', base: BASE });
  return oReady;
}
async function ppRead(rgba: ArrayBuffer, width: number, height: number): Promise<{ words: OcrWord[]; ms: number; device: string }> {
  await loadPpReader();
  const id = ++oId;
  return new Promise((res, rej) => {
    oPending.set(id, { res, rej });
    ow!.postMessage({ type: 'read', id, rgba, width, height }, [rgba]);
  });
}

// ---------------- Florence-2 second reader ----------------
let fw: Worker | null = null;
let fReady: Promise<string> | null = null;
let reqId = 0;
const pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();

export function loadSecondReader(onProgress: (loaded: number, total: number) => void, onDevice: (d: string) => void): Promise<string> {
  if (fReady) return fReady;
  fw = new Worker(new URL('./florence.worker.ts', import.meta.url), { type: 'module' });
  fReady = new Promise((res, rej) => {
    fw!.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') onProgress(m.loaded, m.total);
      else if (m.type === 'device') onDevice(m.device);
      else if (m.type === 'ready') res(m.device);
      else if (m.type === 'result') pending.get(m.id)?.res(m), pending.delete(m.id);
      else if (m.type === 'error') {
        if (m.id !== undefined && pending.has(m.id)) pending.get(m.id)!.rej(new Error(m.error)), pending.delete(m.id);
        else { fReady = null; rej(new Error(m.error)); }
      }
    };
  });
  fw.postMessage({ type: 'load', base: BASE });
  return fReady;
}

export async function secondRead(doc: DocResult): Promise<{ regions: SecondReading[]; ms: number }> {
  if (!fw) throw new Error('Second reader not loaded');
  const id = ++reqId;
  // documents taller than wide get split into tiles so small print has enough pixels
  const portrait = doc.grayH > doc.grayW * 1.1;
  const tiles = portrait ? { cols: 1, rows: 2 } : { cols: 1, rows: 1 };
  const rgba = doc.grayRgba.slice(0);
  const r: any = await new Promise((res, rej) => {
    pending.set(id, { res, rej });
    fw!.postMessage({ type: 'read', id, rgba, width: doc.grayW, height: doc.grayH, tiles }, [rgba]);
  });
  const s = doc.ocrScale;
  const regions: SecondReading[] = r.regions.map((x: SecondReading) => ({ text: x.text, box: x.box.map((v) => v / s) as any }));
  return { regions, ms: r.ms };
}

export async function reanalyse(doc: DocResult, second?: SecondReading[]): Promise<Analysis> {
  if (doc.template) return templateAnalysis(doc.template, doc);
  return analyse(doc.words, doc.width, doc.height, await classifiers(), second, doc.kind, doc.reader);
}

// ---------------- vision critic (MobileCLIP image encoder) ----------------
let cw: Worker | null = null;
let cReady: Promise<void> | null = null;
let cId = 0;
const cPending = new Map<number, { res: (v: VisionOpinion) => void; rej: (e: any) => void }>();

export function loadCritic(onProgress?: (loaded: number, total: number) => void): Promise<void> {
  if (cReady) return cReady;
  cw = new Worker(new URL('./critic.worker.ts', import.meta.url), { type: 'module' });
  cReady = new Promise((res, rej) => {
    cw!.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') onProgress?.(m.loaded, m.total);
      else if (m.type === 'ready') res();
      else if (m.type === 'result') cPending.get(m.id)?.res(m.opinion), cPending.delete(m.id);
      else if (m.type === 'error') {
        if (m.id !== undefined && cPending.has(m.id)) cPending.get(m.id)!.rej(new Error(m.error)), cPending.delete(m.id);
        else { cReady = null; rej(new Error(m.error)); }
      }
    };
  });
  cw.postMessage({ type: 'load', base: BASE });
  return cReady;
}

/** Ask the vision critic what the page looks like (it sees pixels only, never the OCR text). */
export async function criticLook(doc: DocResult): Promise<VisionOpinion> {
  await loadCritic();
  const img = new Image();
  img.src = doc.view;
  await img.decode();
  const k = Math.min(1, 512 / Math.max(img.width, img.height));
  const c = canvas(Math.round(img.width * k), Math.round(img.height * k));
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const rgba = ctx.getImageData(0, 0, c.width, c.height).data.buffer as ArrayBuffer;
  const id = ++cId;
  return new Promise((res, rej) => {
    cPending.set(id, { res, rej });
    cw!.postMessage({ type: 'look', id, rgba, width: c.width, height: c.height }, [rgba]);
  });
}
