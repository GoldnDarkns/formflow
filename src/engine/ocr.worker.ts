/// <reference lib="webworker" />
// PP-OCRv5 reader in its own worker, so detection + recognition never block the interface.
// The Arabic recogniser (8 MB) is loaded only the first time an identity document asks for it.
import * as ort from 'onnxruntime-web/webgpu';
import { loadPpOcr, loadRecogniser, type PpOcr, type Recogniser } from '../core/ppocr';

let reader: Promise<PpOcr> | null = null;
let arabic: Promise<Recogniser> | null = null;
let device = 'wasm';
const post = (m: any) => (self as unknown as Worker).postMessage(m);
const OPTS = { executionProviders: ['wasm'] };

async function load(base: string): Promise<PpOcr> {
  ort.env.wasm.wasmPaths = base + 'ort/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const dict = await (await fetch(base + 'models/ppocr/dict.txt')).text();
  const urls = [base + 'models/ppocr/det.onnx', base + 'models/ppocr/rec.onnx'] as const;
  // WASM only: the WebGPU path for these two models has not been verified on real devices yet
  return loadPpOcr(ort as any, urls[0], urls[1], dict, OPTS);
}
// crops 1.5 times wider than the English reader's: measured on made-up Arabic lines (scripts/eval-arabic.ts)
const loadArabic = async (base: string) => loadRecogniser(ort as any, base + 'models/ppocr/arabic_rec.onnx', await (await fetch(base + 'models/ppocr/arabic_dict.txt')).text(), { ...OPTS, logSeverityLevel: 3 }, { widen: 1.5 });

self.onmessage = async (e: MessageEvent<any>) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      reader ??= load(m.base);
      await reader;
      post({ type: 'ready', device });
    } else if (m.type === 'read') {
      const r = await reader!;
      const t = performance.now();
      let ar: Recogniser | undefined;
      if (m.arabic) {
        arabic ??= loadArabic(m.base);
        ar = await arabic.catch(() => { arabic = null; return undefined; }); // without the Arabic model the English reading still comes back
      }
      const { words, arabic: lines } = await r.readAll(new Uint8ClampedArray(m.rgba), m.width, m.height, ar);
      post({ type: 'result', id: m.id, words, arabic: lines, arabicRead: !!ar, ms: performance.now() - t, device });
    }
  } catch (err: any) {
    if (m.type === 'load') reader = null;
    post({ type: 'error', id: m.id, error: String(err?.message ?? err) });
  }
};
