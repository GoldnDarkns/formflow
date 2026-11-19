/// <reference lib="webworker" />
// PP-OCRv5 reader in its own worker, so detection + recognition never block the interface.
import * as ort from 'onnxruntime-web/webgpu';
import { loadPpOcr, type PpOcr } from '../core/ppocr';

let reader: Promise<PpOcr> | null = null;
let device = 'wasm';
const post = (m: any) => (self as unknown as Worker).postMessage(m);

async function load(base: string): Promise<PpOcr> {
  ort.env.wasm.wasmPaths = base + 'ort/';
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const dict = await (await fetch(base + 'models/ppocr/dict.txt')).text();
  const urls = [base + 'models/ppocr/det.onnx', base + 'models/ppocr/rec.onnx'] as const;
  // WASM only: the WebGPU path for these two models has not been verified on real devices yet
  return loadPpOcr(ort as any, urls[0], urls[1], dict, { executionProviders: ['wasm'] });
}

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
      const words = await r.read(new Uint8ClampedArray(m.rgba), m.width, m.height);
      post({ type: 'result', id: m.id, words, ms: performance.now() - t, device });
    }
  } catch (err: any) {
    if (m.type === 'load') reader = null;
    post({ type: 'error', id: m.id, error: String(err?.message ?? err) });
  }
};
