/// <reference lib="webworker" />
// Florence-2 second reader in its own worker (model download + inference).
import { env, RawImage } from '@huggingface/transformers';
import { loadFlorence, type FlorenceReader } from '../core/florence';

env.allowLocalModels = false;

let reader: FlorenceReader | null = null;
let loading: Promise<FlorenceReader> | null = null;

async function pickDevice(): Promise<'webgpu' | 'wasm'> {
  try {
    const gpu = (navigator as any).gpu;
    if (gpu && (await gpu.requestAdapter())) return 'webgpu';
  } catch {}
  return 'wasm';
}

const post = (m: any) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (e: MessageEvent<any>) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      if (!loading) {
        (env.backends.onnx as any).wasm.wasmPaths = msg.base + 'ort/';
        const device = await pickDevice();
        post({ type: 'device', device });
        const files = new Map<string, { loaded: number; total: number }>();
        const progress = (p: { file: string; loaded: number; total: number }) => {
          files.set(p.file, p);
          let l = 0, t = 0;
          for (const v of files.values()) { l += v.loaded; t += v.total; }
          post({ type: 'progress', loaded: l, total: t });
        };
        // Safari on iPhone now reports a graphics card, but ONNX Runtime cannot always start on it
        // ("no available backend found"). Then the processor is tried instead, and if that fails too
        // the phone has not given the page enough memory for a 275 MB model.
        loading = loadFlorence(device, progress).catch(async (first) => {
          if (device !== 'webgpu') throw first;
          post({ type: 'device', device: 'wasm' });
          files.clear();
          return loadFlorence('wasm', progress);
        }).catch((err) => {
          throw new Error(/backend|memory|allocat|RangeError/i.test(String(err?.message ?? err))
            ? 'This device would not start Florence-2 (it needs about 1 GB of memory for the page). It runs on a laptop. Everything else works without it.'
            : String(err?.message ?? err));
        });
      }
      reader = await loading;
      post({ type: 'ready', device: reader.device });
    } else if (msg.type === 'read') {
      if (!reader) reader = await loading!;
      const t = performance.now();
      const img = new RawImage(new Uint8ClampedArray(msg.rgba), msg.width, msg.height, 4).rgb();
      const regions = await reader.read(img, msg.tiles);
      post({ type: 'result', id: msg.id, regions, ms: performance.now() - t });
    }
  } catch (err: any) {
    loading = null;
    post({ type: 'error', id: msg.id, error: String(err?.message ?? err) });
  }
};
