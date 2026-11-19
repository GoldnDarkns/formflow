// The critic's eyes: MobileCLIP-S0's image encoder (about 46 MB) looks at the
// picture itself, not the OCR text, and compares it with a description of each
// document type. The descriptions are embedded once, offline, by the matching
// text encoder (scripts/critic-embed.ts) and shipped as a small JSON file, so
// the browser only downloads the image encoder. Because this model never sees
// the sorter's evidence, agreement between the two is a real second opinion.
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from '@huggingface/transformers';
import type { DocType } from './classify';

export const CRITIC_ID = 'Xenova/mobileclip_s0';
export const PROMPTS: [DocType, string[]][] = [
  ['passport', ['a photo of a passport data page', 'an open passport showing the photo page']],
  ['emirates_id', ['a photo of a United Arab Emirates identity card', 'an Emirates ID card']],
  ['driving_licence', ['a photo of a driving licence card', "a driver's license"]],
  ['visa', ['a visa or entry permit document', 'an e-visa approval page']],
  ['form', ['a scanned paper form or office document full of text', 'a fax or printed business form']],
  ['other', ['a photo of something that is not a document', 'a random photograph']],
];

export interface TextEmbeds { types: DocType[]; dim: number; embeds: number[][] }
export interface VisionOpinion {
  type: DocType;
  confidence: number;
  probs: Partial<Record<DocType, number>>;
}
export interface VisionCritic {
  look(img: RawImage): Promise<VisionOpinion>;
}

export async function loadVisionCritic(device: 'webgpu' | 'wasm' | 'cpu', text: TextEmbeds, onProgress?: (p: { file: string; loaded: number; total: number }) => void, localPath?: string): Promise<VisionCritic> {
  if (localPath) { env.localModelPath = localPath; env.allowRemoteModels = false; }
  const progress_callback = (p: any) => { if (p.status === 'progress' && onProgress) onProgress({ file: p.file, loaded: p.loaded, total: p.total }); };
  // full-precision weights: the 8-bit version of this model loses its zero-shot ability
  const [proc, vision] = await Promise.all([
    AutoProcessor.from_pretrained(CRITIC_ID, { progress_callback } as any),
    CLIPVisionModelWithProjection.from_pretrained(CRITIC_ID, { dtype: 'fp32', device, progress_callback } as any),
  ]);
  return {
    async look(img) {
      const v = await vision(await (proc as any)(img));
      const raw = v.image_embeds.data as Float32Array;
      const n = Math.hypot(...raw);
      const logits = text.embeds.map((tv) => 100 * tv.reduce((a, x, i) => a + (x * raw[i]) / n, 0));
      const mx = Math.max(...logits);
      const ex = logits.map((l) => Math.exp(l - mx));
      const Z = ex.reduce((a, b) => a + b, 0);
      const probs: Partial<Record<DocType, number>> = {};
      text.types.forEach((t, i) => (probs[t] = ex[i] / Z));
      const k = ex.indexOf(Math.max(...ex));
      return { type: text.types[k], confidence: ex[k] / Z, probs };
    },
  };
}
