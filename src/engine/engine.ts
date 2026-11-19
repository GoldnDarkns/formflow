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

import { believableMrz, linesFromWords, mrzCrops, mrzNeuralCrops, readingQuality, TURNS, TURN_GAIN, UPRIGHT_MIN } from '../core/orient';
import { matchTemplate, readTemplate, templateChecks, templateWords, type Template, type TemplateMatch } from '../core/template';
import { checkState, checkboxLabel, dropLetterBoxes, findCheckboxes, findSignatures, isSignatureCaption } from '../core/marks';
import type { Box, Field } from '../core/types';
import { rgbaToGray } from '../core/preprocess';
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
  source: Blob; // the file as given, kept in memory so the page can be read again with corrected corners
  manualQuad?: boolean; // the corners were set by hand
  turned: number; // degrees the photo was turned to read it upright (0, 90, 180, 270)
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
// ---------------- checkboxes and signatures ----------------
interface MarkInput { kind: 'form' | 'id'; words: OcrWord[]; width: number; height: number; grayRgba: ArrayBuffer; grayW: number; grayH: number; ocrScale: number }
export const TICKED = 'Ticked', NOT_TICKED = 'Not ticked', SIGNED = 'Signed', NOT_SIGNED = 'Not signed';
/**
 * Add what is not text to a form's fields: each checkbox with its state and each signing
 * line with "signed / not signed". With a template the places are known; without one the
 * boxes are found on the page and labelled with the text printed beside them.
 */
function addMarks(a: Analysis, d: MarkInput, tm?: TemplateMatch | null): Analysis {
  if (d.kind !== 'form') return a;
  try {
    const g = rgbaToGray(new Uint8ClampedArray(d.grayRgba), d.grayW, d.grayH);
    const s = d.ocrScale;
    const up = (b: Box) => b.map((v) => v * s) as Box, down = (b: Box) => b.map((v) => v / s) as Box;
    const cx = (b: Box) => (b[0] + b[2]) / 2, cy = (b: Box) => (b[1] + b[3]) / 2;
    const entities = [...a.entities], links = [...a.links], fields = [...a.fields];
    const push = (key: string, value: string, box: Box, confidence: number, reasons: string[] = []) => {
      const qId = entities.length;
      entities.push({ id: qId, text: key, box: [box[0], box[1], box[0], box[1]], words: [], label: 'question', probs: [1, 0, 0, 0] });
      const aId = entities.length;
      entities.push({ id: aId, text: value, box, words: [], label: 'answer', probs: [0, 1, 0, 0] });
      links.push({ q: qId, a: aId, p: confidence });
      const f: Field = { key, value, qId, aId, confidence, raw: confidence, status: reasons.length ? 'review' : 'ok', reasons, source: 'mark' };
      fields.push(f);
    };
    // how sure a tick reading is: far from the threshold = sure
    const tickConf = (ink: number) => (ink > 0.1 ? Math.min(0.99, 0.75 + ink) : ink < 0.04 ? 0.98 : 0.7);
    const found = findCheckboxes(g);
    if (tm?.template.checks?.length) {
      for (const c of templateChecks(tm, d.width, d.height)) {
        const b = up(c.box), size = Math.max(b[2] - b[0], b[3] - b[1]);
        const near = found.find((f) => Math.hypot(cx(f.box) - cx(b), cy(f.box) - cy(b)) < size * 0.6);
        const st = near ?? checkState(g, b);
        push(c.label, st.checked ? TICKED : NOT_TICKED, near ? down(near.box) : c.box, tickConf(st.ink));
      }
    } else {
      const real = dropLetterBoxes(found, d.words, s);
      const boxes = real.map((f) => down(f.box));
      real.forEach((f, i) => {
        const label = checkboxLabel(boxes[i], d.words, boxes);
        if (label.length >= 2) push(label, f.checked ? TICKED : NOT_TICKED, boxes[i], tickConf(f.ink), f.ink > 0.06 && f.ink <= 0.1 ? ['Faint mark in the box'] : []);
      });
    }
    const caps = tm ? templateWords(tm, d.width, d.height, isSignatureCaption) : undefined;
    for (const sig of findSignatures(g, d.words, s, caps?.length ? caps : undefined)) {
      // measured on the 20-form set: right 88% of the time with a template, 73% without
      push('Signature', sig.signed ? SIGNED : NOT_SIGNED, sig.box ?? sig.caption, tm ? 0.88 : 0.73, sig.signed ? [] : ['No pen stroke found by the signature line']);
    }
    return { ...a, entities, links, fields };
  } catch (e) {
    console.warn('marks', e);
    return a;
  }
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
  const analysis = addMarks(templateAnalysis(m, doc), { ...doc, kind: 'form' }, m);
  return { ...doc, kind: 'form', docType: 'form', template: m, analysis, classification: templateClass(doc.classification, m), facts: extractFacts('form', doc.words, analysis.fields, doc.mrz) };
}

export function classifiers(): Promise<Classifiers> {
  return (clf ??= loadClassifiers(ort as any, BASE + 'models/entity_gbdt.onnx', BASE + 'models/link_gbdt.onnx', { executionProviders: ['wasm'] }));
}
/** warm up engines in the background so the first scan is fast */
export function warmUp() {
  tesseract().catch(() => (tess = null));
  classifiers().catch(() => (clf = null));
  // the neural reader is used for passport and ID code lines on every page, whichever main reader is chosen
  loadPpReader().catch(() => {});
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
 * Read the code lines from several crops (where the first pass saw rows of "<",
 * the bottom quarter, the bottom 40%; each from the grey page and from the
 * cleaned black-and-white page) and keep the reading whose check digits agree
 * best. Stops as soon as every check digit passes, so a clean passport costs
 * one extra read and only hard pages cost more.
 */
async function readMrz(gray: HTMLCanvasElement, clean: HTMLCanvasElement, band: [number, number] | null, nWords: number, grayRgba?: ArrayBuffer): Promise<MrzResult | null> {
  const crops = mrzCrops(band, gray.height, nWords);
  if (!crops.length) return null;
  let best: MrzResult | null = null;
  // the neural reader first: it is far better at these lines; the classical crops below are the fallback
  if (grayRgba) {
    try {
      const W = gray.width;
      for (const [y0, y1] of mrzNeuralCrops(band, gray.height, nWords)) {
        const part = new Uint8ClampedArray(grayRgba).slice(y0 * W * 4, y1 * W * 4);
        const r = await ppRead(part.buffer as ArrayBuffer, W, y1 - y0);
        const lines = findMrzLines(linesFromWords(r.words));
        const m = lines.length >= 2 ? parseMrz(lines) : null;
        if (believableMrz(m, !!band) && mrzScore(m) > mrzScore(best)) best = m;
        if (best?.valid) return best;
      }
    } catch (e) {
      console.warn('neural code-line read unavailable, using the classical reader', e);
    }
  }
  const w = await tesseractMrz();
  for (const cr of crops) {
    const src = cr.binarised ? clean : gray;
    const h = Math.max(1, cr.y1 - cr.y0);
    const c = canvas(src.width, h);
    c.getContext('2d')!.drawImage(src, 0, cr.y0, src.width, h, 0, 0, src.width, h);
    const { data } = await w.recognize(c);
    const lines = findMrzLines(data.text);
    const m = lines.length >= 2 ? parseMrz(lines) : null;
    if (believableMrz(m, !!band) && mrzScore(m) > mrzScore(best)) best = m;
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

export async function processDocument(file: Blob, fileName: string, opts: PreprocessOptions, onStep: (s: Step[]) => void, onLive?: (l: Live) => void, manualQuad?: Quad, turn = 0): Promise<DocResult> {
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
  let original = shrink(src, 900).toDataURL('image/jpeg', 0.7);
  onLive?.({ original });
  const rgba = sctx.getImageData(0, 0, W0, H0).data.buffer as ArrayBuffer;
  upd(0, { state: 'done', ms: performance.now() - t, detail: `${W0}×${H0}` });

  upd(1, { state: 'running' });
  const tessP = tesseract();
  const clfP = classifiers();
  // one full reading of the page as it stands: find it, flatten it, clean it, read it
  const readPage = async (buf: ArrayBuffer, w0: number, h0: number, first: boolean) => {
    // corners dragged by the person (0..1 of the photo) replace the automatic page finder
    const p = await runPre({ rgba: buf, width: w0, height: h0, opts, quad: manualQuad ? (manualQuad.map(([x, y]) => [x * w0, y * h0]) as Quad) : undefined });
    const W = p.baseW, H = p.baseH;
    const quad = p.quad ? (p.quad.map(([x, y]) => [x / w0, y / h0]) as Quad) : null;
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
    if (first) {
      onLive?.({ quad, flat: viewUrl });
      upd(1, { state: 'done', ms: p.ms, detail: [quad ? 'page cut out' : '', p.lit ? 'lighting fixed' : '', p.skew ? `straightened ${Math.abs(p.skew).toFixed(1)}°` : ''].filter(Boolean).join(', ') || 'already clean' });
      // give the cut-out animation a moment on screen when a page was found
      if (quad) await new Promise((r) => setTimeout(r, 900));
      t = performance.now();
      upd(2, { state: 'running' });
    }
    const ocrCanvas = rgbaToCanvas(p.ocr, p.width, p.height);
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
    return { p, W, H, quad, viewUrl, ocrCanvas, words, readBy };
  };
  const turnedCanvas = (deg: number) => {
    const swap = deg % 180 !== 0;
    const c = canvas(swap ? H0 : W0, swap ? W0 : H0);
    const x = c.getContext('2d', { willReadFrequently: true })!;
    x.translate(c.width / 2, c.height / 2);
    x.rotate((deg * Math.PI) / 180);
    x.drawImage(src, -W0 / 2, -H0 / 2);
    return c;
  };
  let turned = turn;
  let page: Awaited<ReturnType<typeof readPage>>;
  if (turn) {
    // the caller already knows which way up this page goes (re-reading with corrected corners)
    const c = turnedCanvas(turn);
    original = shrink(c, 900).toDataURL('image/jpeg', 0.7);
    page = await readPage(c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data.buffer as ArrayBuffer, c.width, c.height, true);
  } else {
    page = await readPage(rgba, W0, H0, true);
    // photographed sideways or upside down? almost no clean words come out, so turn it and read again
    if (readingQuality(page.words) <= UPRIGHT_MIN && !manualQuad) {
      for (const deg of TURNS) {
        const c = turnedCanvas(deg);
        const alt = await readPage(c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data.buffer as ArrayBuffer, c.width, c.height, false);
        if (readingQuality(alt.words) >= readingQuality(page.words) + TURN_GAIN) {
          page = alt; turned = deg;
          original = shrink(c, 900).toDataURL('image/jpeg', 0.7);
          onLive?.({ original, quad: alt.quad, flat: alt.viewUrl });
          if (readingQuality(alt.words) >= 8) break;
        }
      }
    }
  }
  const { p, W, H, quad, viewUrl, ocrCanvas, words } = page;
  const readBy = page.readBy + (turned ? `, turned ${turned}°` : '');
  upd(2, { state: 'done', ms: performance.now() - t, detail: `${words.length} words, ${readBy}` });

  // machine-readable zone first: it is the strongest clue to the document type
  t = performance.now();
  upd(3, { state: 'running' });
  let mrz: MrzResult | null = null;
  const band = mrzBand(words, p.scale, p.height);
  mrz = await readMrz(rgbaToCanvas(p.gray, p.width, p.height), ocrCanvas, band, words.length, p.gray);
  let cls = classifyDocument(words, W, H, mrz);
  // a page whose printed words line up with a known form is that form, whatever the sorter thought
  const tm = mrz?.valid ? null : matchTemplate(await templates(), words, W, H);
  if (tm) cls = templateClass(cls, tm);
  upd(3, { state: 'done', ms: performance.now() - t, detail: `${DOC_LABEL[cls.type]}, ${Math.round(cls.confidence * 100)}% sure` });

  t = performance.now();
  upd(4, { state: 'running' });
  const kind = ID_TYPES.includes(cls.type) ? 'id' : 'form';
  const c = await clfP;
  const analysis = addMarks(
    tm ? templateAnalysis(tm, { words, width: W, height: H }) : await analyse(words, W, H, c, undefined, kind, opts.reader === 'ppocr' ? 'ppocr' : 'tesseract'),
    { kind, words, width: W, height: H, grayRgba: p.gray, grayW: p.width, grayH: p.height, ocrScale: p.scale }, tm);
  const facts = extractFacts(cls.type, words, analysis.fields, mrz);
  upd(4, { state: 'done', ms: performance.now() - t, detail: tm ? `${analysis.fields.filter((f) => f.value).length} of ${analysis.fields.length} boxes filled, template: ${tm.template.name}` : `${analysis.fields.filter((f) => f.value).length} fields` });

  return {
    source: file,
    manualQuad: !!manualQuad,
    turned,
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
  const analysis = addMarks(await analyse(doc.words, doc.width, doc.height, await classifiers(), second, kind, doc.reader), { ...doc, kind });
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
  if (doc.template) return addMarks(templateAnalysis(doc.template, doc), doc, doc.template);
  return addMarks(await analyse(doc.words, doc.width, doc.height, await classifiers(), second, doc.kind, doc.reader), doc);
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
