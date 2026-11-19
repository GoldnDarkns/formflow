/// <reference lib="webworker" />
import { env, RawImage } from '@huggingface/transformers';
import { loadVisionCritic, type VisionCritic } from '../core/critic-vision';

env.allowLocalModels = false;
let critic: Promise<VisionCritic> | null = null;
const post = (m: any) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (e: MessageEvent<any>) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      (env.backends.onnx as any).wasm.wasmPaths = msg.base + 'ort/';
      const files = new Map<string, { loaded: number; total: number }>();
      critic ??= fetch(msg.base + 'models/critic-text.json').then((r) => r.json()).then((text) =>
        loadVisionCritic('wasm', text, (p) => {
          files.set(p.file, p);
          let l = 0, t = 0;
          for (const v of files.values()) { l += v.loaded; t += v.total; }
          post({ type: 'progress', loaded: l, total: t });
        }));
      await critic;
      post({ type: 'ready' });
    } else if (msg.type === 'look') {
      const c = await critic!;
      const opinion = await c.look(new RawImage(new Uint8ClampedArray(msg.rgba), msg.width, msg.height, 4).rgb());
      post({ type: 'result', id: msg.id, opinion });
    }
  } catch (err: any) {
    if (msg.type === 'load') critic = null;
    post({ type: 'error', id: msg.id, error: String(err?.message ?? err) });
  }
};
