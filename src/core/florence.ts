// Florence-2 (open vision-language model, MIT licence) as a second reader.
// Runs through Transformers.js: WebGPU or WASM in the browser, onnxruntime in Node.
import { AutoProcessor, Florence2ForConditionalGeneration, RawImage, env } from '@huggingface/transformers';
import type { Box, SecondReading } from './types';

export const FLORENCE_ID = 'onnx-community/Florence-2-base-ft';
const TASK = '<OCR_WITH_REGION>';

type Device = 'webgpu' | 'wasm' | 'cpu';

export interface FlorenceReader {
  /** one-sentence description of what the image shows (used by the critic) */
  caption(img: RawImage, task?: string): Promise<string>;
  read(img: RawImage, tiles?: { cols: number; rows: number }): Promise<SecondReading[]>;
  device: Device;
}

export function dtypesFor(device: Device) {
  // int8 weights on CPU/WASM (~275 MB, cached after first download); fp16 vision + q4 text on WebGPU
  return device === 'webgpu'
    ? { embed_tokens: 'fp16', vision_encoder: 'fp16', encoder_model: 'q4', decoder_model_merged: 'q4' }
    : { embed_tokens: 'q8', vision_encoder: 'q8', encoder_model: 'q8', decoder_model_merged: 'q8' };
}

export async function loadFlorence(device: Device, onProgress?: (p: { file: string; loaded: number; total: number }) => void, localPath?: string): Promise<FlorenceReader> {
  if (localPath) {
    env.localModelPath = localPath;
    env.allowRemoteModels = false;
  }
  const progress_callback = (p: any) => {
    if (p.status === 'progress' && onProgress) onProgress({ file: p.file, loaded: p.loaded, total: p.total });
  };
  const model = await Florence2ForConditionalGeneration.from_pretrained(FLORENCE_ID, {
    dtype: dtypesFor(device) as any,
    device: device as any,
    progress_callback,
  });
  const processor: any = await AutoProcessor.from_pretrained(FLORENCE_ID, { progress_callback } as any);
  const prompts = processor.construct_prompts(TASK);

  async function readOne(img: RawImage, offX: number, offY: number): Promise<SecondReading[]> {
    const inputs = await processor(img, prompts);
    const ids: any = await model.generate({ ...inputs, max_new_tokens: 1024, num_beams: 1, do_sample: false } as any);
    const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
    const res = processor.post_process_generation(text, TASK, [img.width, img.height])[TASK];
    const out: SecondReading[] = [];
    res.labels.forEach((label: string, i: number) => {
      const q: number[] = res.quad_boxes[i];
      const xs = [q[0], q[2], q[4], q[6]], ys = [q[1], q[3], q[5], q[7]];
      const t = label.replace(/<\/?s>|<pad>/g, '').trim();
      if (!t) return;
      out.push({ text: t, box: [Math.min(...xs) + offX, Math.min(...ys) + offY, Math.max(...xs) + offX, Math.max(...ys) + offY] });
    });
    return out;
  }

  return {
    device,
    async caption(img, task = '<DETAILED_CAPTION>') {
      const inputs = await processor(img, processor.construct_prompts(task));
      const ids: any = await model.generate({ ...inputs, max_new_tokens: 56, num_beams: 1, do_sample: false } as any);
      const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
      return String(processor.post_process_generation(text, task, [img.width, img.height])[task]).replace(/<\/?s>|<pad>/g, '').trim();
    },
    async read(img, tiles = { cols: 1, rows: 1 }) {
      if (tiles.cols === 1 && tiles.rows === 1) return readOne(img, 0, 0);
      // overlapping tiles so small print gets enough pixels at the model's 768x768 input
      const out: SecondReading[] = [];
      const tw = img.width / tiles.cols, th = img.height / tiles.rows;
      const ox = tw * 0.08, oy = th * 0.08;
      for (let r = 0; r < tiles.rows; r++)
        for (let c = 0; c < tiles.cols; c++) {
          const x0 = Math.max(0, Math.round(c * tw - ox)), y0 = Math.max(0, Math.round(r * th - oy));
          const x1 = Math.min(img.width, Math.round((c + 1) * tw + ox)), y1 = Math.min(img.height, Math.round((r + 1) * th + oy));
          const crop = await img.crop([x0, y0, x1 - 1, y1 - 1]);
          out.push(...(await readOne(crop, x0, y0)));
        }
      return dedupe(out);
    },
  };
}

/** drop duplicates produced in tile overlaps (same text, overlapping boxes) */
function dedupe(rs: SecondReading[]): SecondReading[] {
  const out: SecondReading[] = [];
  const area = (b: Box) => (b[2] - b[0]) * (b[3] - b[1]);
  for (const r of rs) {
    const dup = out.findIndex((o) => {
      const ix = Math.max(0, Math.min(o.box[2], r.box[2]) - Math.max(o.box[0], r.box[0]));
      const iy = Math.max(0, Math.min(o.box[3], r.box[3]) - Math.max(o.box[1], r.box[1]));
      return (ix * iy) / Math.max(1, Math.min(area(o.box), area(r.box))) > 0.6;
    });
    if (dup < 0) out.push(r);
    else if (r.text.length > out[dup].text.length) out[dup] = r;
  }
  return out;
}
