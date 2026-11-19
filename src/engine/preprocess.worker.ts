/// <reference lib="webworker" />
// Image work runs off the main thread so the interface stays smooth:
// find the page -> flatten it -> even out the light -> deskew -> binarise.
import { preprocess, resize, rgbaToGray, grayToRgba, ocrScale, backgroundUnevenness, type PreprocessOptions } from '../core/preprocess';
import { detectPage, flattenLighting, warpQuad, type Quad } from '../core/scan';

export interface PreRequest {
  rgba: ArrayBuffer;
  width: number;
  height: number;
  opts: PreprocessOptions;
}
export interface PreResponse {
  color: ArrayBuffer; // RGBA of the flattened, evenly lit page (baseW x baseH)
  baseW: number;
  baseH: number;
  quad: Quad | null; // page corners found in the original photo
  lit: boolean; // lighting was evened out
  ocr: ArrayBuffer; // RGBA of the cleaned image fed to Tesseract
  gray: ArrayBuffer; // RGBA of deskewed + upscaled grayscale (for MRZ crops)
  width: number;
  height: number;
  scale: number;
  skew: number;
  ms: number;
}

self.onmessage = (e: MessageEvent<PreRequest>) => {
  const t = performance.now();
  const { opts } = e.data;
  let color = new Uint8ClampedArray(e.data.rgba);
  let W = e.data.width, H = e.data.height;
  let quad: Quad | null = null;
  let lit = false;
  if (opts.magic) {
    const g0 = rgbaToGray(color, W, H);
    quad = detectPage(g0);
    if (quad) {
      const w = warpQuad(color, W, H, quad);
      color = w.data as Uint8ClampedArray<ArrayBuffer>;
      W = w.width;
      H = w.height;
    }
    // camera photos (a page was cut out) and unevenly lit scans get their lighting fixed
    if (quad || backgroundUnevenness(quad ? rgbaToGray(color, W, H) : g0) > 12) {
      flattenLighting(color, W, H);
      lit = true;
    }
  }
  let g = rgbaToGray(color, W, H);
  const scale = ocrScale(W, H);
  g = resize(g, scale);
  const deskewOnly = preprocess(g, { ...opts, deskew: opts.deskew, binarise: false, removeLines: false });
  const full = opts.binarise ? preprocess(deskewOnly.img, { ...opts, deskew: false, binarise: true, removeLines: opts.removeLines }).img : deskewOnly.img;
  const ocr = grayToRgba(full).buffer as ArrayBuffer;
  const gray = grayToRgba(deskewOnly.img).buffer as ArrayBuffer;
  const res: PreResponse = { color: color.buffer as ArrayBuffer, baseW: W, baseH: H, quad, lit, ocr, gray, width: g.width, height: g.height, scale, skew: deskewOnly.skew, ms: performance.now() - t };
  (self as unknown as Worker).postMessage(res, [res.color, ocr, gray]);
};
