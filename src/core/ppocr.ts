// PP-OCRv5 (mobile) as an alternative text reader: a neural text detector
// (DBNet-style, 4.6 MB) finds text segments, a neural recogniser (SVTR-style,
// 7.5 MB, English) reads each one with CTC decoding. Both are plain ONNX and
// run with ONNX Runtime in the browser (WASM) or in Node. Apache-2.0.
import type { Box, OcrWord } from './types';

interface OrtLike {
  InferenceSession: { create(src: any, opts?: any): Promise<any> };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => any;
}
export interface PpOcr {
  read(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Promise<OcrWord[]>;
}
export interface PpOptions {
  maxSide?: number; // detector input size (long side)
  boxThresh?: number;
  unclip?: number;
}

/** bilinear resize of an RGBA region into a planar float RGB tensor, normalised (v/255 - mean) / std */
function toTensor(rgba: Uint8ClampedArray | Uint8Array, w: number, sx: number, sy: number, sw: number, sh: number, tw: number, th: number, padW: number, mean: number[], std: number[]): Float32Array {
  const out = new Float32Array(3 * th * padW);
  const fx = sw / tw, fy = sh / th;
  for (let y = 0; y < th; y++) {
    const yy = Math.min(sy + sh - 1.001, Math.max(sy, sy + (y + 0.5) * fy - 0.5)), y0 = yy | 0, wy = yy - y0;
    for (let x = 0; x < tw; x++) {
      const xx = Math.min(sx + sw - 1.001, Math.max(sx, sx + (x + 0.5) * fx - 0.5)), x0 = xx | 0, wx = xx - x0;
      const i = (y0 * w + x0) * 4;
      for (let c = 0; c < 3; c++) {
        const v = rgba[i + c] * (1 - wx) * (1 - wy) + rgba[i + 4 + c] * wx * (1 - wy) + rgba[i + w * 4 + c] * (1 - wx) * wy + rgba[i + w * 4 + 4 + c] * wx * wy;
        out[c * th * padW + y * padW + x] = (v / 255 - mean[c]) / std[c];
      }
    }
  }
  return out;
}

export async function loadPpOcr(ort: OrtLike, detSrc: any, recSrc: any, dictText: string, sessionOpts?: any, opt: PpOptions = {}): Promise<PpOcr> {
  const [det, rec] = await Promise.all([ort.InferenceSession.create(detSrc, sessionOpts), ort.InferenceSession.create(recSrc, sessionOpts)]);
  const chars = ['', ...dictText.split(/\r?\n/).filter((l, i, a) => !(i === a.length - 1 && l === '')), ' '];
  // unclip 0.8 was chosen on 12 FUNSD training pages: on dense forms a larger growth makes neighbouring lines bleed into each crop
  const maxSide = opt.maxSide ?? 1600, boxThresh = opt.boxThresh ?? 0.55, unclip = opt.unclip ?? 0.8;

  async function detect(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Promise<{ box: Box; score: number }[]> {
    const k = Math.min(1, maxSide / Math.max(w, h));
    const tw = Math.max(32, Math.round((w * k) / 32) * 32), th = Math.max(32, Math.round((h * k) / 32) * 32);
    const input = toTensor(rgba, w, 0, 0, w, h, tw, th, tw, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, th, tw]) });
    const p = out[det.outputNames[0]].data as Float32Array;
    // connected components of the thresholded probability map
    const lab = new Int32Array(tw * th);
    const boxes: { box: Box; score: number }[] = [];
    const stack: number[] = [];
    let cur = 0;
    for (let s = 0; s < tw * th; s++) {
      if (p[s] < 0.3 || lab[s]) continue;
      cur++;
      let x0 = tw, y0 = th, x1 = 0, y1 = 0, sum = 0, n = 0;
      stack.push(s);
      lab[s] = cur;
      while (stack.length) {
        const i = stack.pop()!, x = i % tw, y = (i / tw) | 0;
        sum += p[i]; n++;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && !lab[i - 1] && p[i - 1] >= 0.3) { lab[i - 1] = cur; stack.push(i - 1); }
        if (x < tw - 1 && !lab[i + 1] && p[i + 1] >= 0.3) { lab[i + 1] = cur; stack.push(i + 1); }
        if (y > 0 && !lab[i - tw] && p[i - tw] >= 0.3) { lab[i - tw] = cur; stack.push(i - tw); }
        if (y < th - 1 && !lab[i + tw] && p[i + tw] >= 0.3) { lab[i + tw] = cur; stack.push(i + tw); }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      if (bw < 3 || bh < 3 || sum / n < boxThresh) continue;
      // the detector predicts a shrunk text kernel: grow it back (DB "unclip")
      const d = (bw * bh * unclip) / (2 * (bw + bh));
      boxes.push({ score: sum / n, box: [Math.max(0, (x0 - d) / tw * w), Math.max(0, (y0 - d) / th * h), Math.min(w, (x1 + 1 + d) / tw * w), Math.min(h, (y1 + 1 + d) / th * h)] });
    }
    return boxes;
  }

  type Rec = { text: string; conf: number; spans: [number, number][] };
  /** Recognise several segments in one run: crops are resized to height 48 and right-padded to the widest. */
  async function recogniseBatch(rgba: Uint8ClampedArray | Uint8Array, w: number, boxes: Box[]): Promise<Rec[]> {
    const tws = boxes.map((b) => Math.max(16, Math.min(1600, Math.round((48 * Math.max(2, b[2] - b[0])) / Math.max(2, b[3] - b[1])))));
    const padW = Math.max(...tws), N = boxes.length;
    const input = new Float32Array(N * 3 * 48 * padW);
    boxes.forEach((b, i) => input.set(toTensor(rgba, w, b[0], b[1], Math.max(2, b[2] - b[0]), Math.max(2, b[3] - b[1]), tws[i], 48, padW, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), i * 3 * 48 * padW));
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [N, 3, 48, padW]) });
    const t = out[rec.outputNames[0]];
    const [, T, C] = t.dims as number[];
    const d = t.data as Float32Array;
    return boxes.map((_, bi) => {
      const frac = tws[bi] / padW; // the part of the time axis that holds real pixels
      let text = '', last = 0, sum = 0, n = 0;
      const spans: [number, number][] = [];
      for (let i = 0; i < T; i++) {
        const o = (bi * T + i) * C;
        let k = 0, m = -Infinity;
        for (let c = 0; c < C; c++) if (d[o + c] > m) { m = d[o + c]; k = c; }
        if (k !== 0 && k !== last && i / T <= frac + 0.02) { text += chars[k] ?? ''; spans.push([Math.min(1, i / T / frac), m]); sum += m; n++; }
        last = k;
      }
      return { text, conf: n ? sum / n : 0, spans };
    });
  }

  return {
    async read(rgba, w, h) {
      const dets = await detect(rgba, w, h);
      // assign text lines by vertical overlap, top to bottom
      dets.sort((a, b) => (a.box[1] + a.box[3]) / 2 - (b.box[1] + b.box[3]) / 2);
      const lineOf: number[] = [];
      let line = -1, ly0 = 0, ly1 = 0;
      dets.forEach((d, i) => {
        const cy = (d.box[1] + d.box[3]) / 2;
        if (line < 0 || cy > ly1 - (ly1 - ly0) * 0.3) { line++; ly0 = d.box[1]; ly1 = d.box[3]; }
        lineOf[i] = line;
      });
      // batch segments of similar width so padding stays small
      const order = dets.map((_, i) => i).sort((a, b) => (dets[a].box[2] - dets[a].box[0]) / (dets[a].box[3] - dets[a].box[1]) - (dets[b].box[2] - dets[b].box[0]) / (dets[b].box[3] - dets[b].box[1]));
      const recs: Rec[] = [];
      for (let s0 = 0; s0 < order.length; ) {
        const ar = (i: number) => (dets[i].box[2] - dets[i].box[0]) / (dets[i].box[3] - dets[i].box[1]);
        let e = s0 + 1;
        while (e < order.length && e - s0 < 12 && ar(order[e]) <= ar(order[s0]) * 1.35 + 0.5) e++;
        const ids = order.slice(s0, e);
        (await recogniseBatch(rgba, w, ids.map((i) => dets[i].box))).forEach((r, k) => (recs[ids[k]] = r));
        s0 = e;
      }
      const words: OcrWord[] = [];
      for (let i = 0; i < dets.length; i++) {
        const { box } = dets[i];
        const r = recs[i];
        if (!r.text.trim()) continue;
        // split the segment into words; x positions come from the CTC time steps
        const W = box[2] - box[0];
        let start = -1, conf = 0, cnt = 0, buf = '';
        const flush = (endT: number) => {
          if (buf.trim()) words.push({ text: buf, conf: (conf / cnt) * 100, line: lineOf[i], box: [box[0] + start * W, box[1], Math.min(box[2], box[0] + endT * W), box[3]] });
          buf = ''; start = -1; conf = 0; cnt = 0;
        };
        const cs = [...r.text];
        cs.forEach((ch, j) => {
          const [t, c] = r.spans[j] ?? [j / cs.length, r.conf];
          if (ch === ' ') return flush(t);
          if (start < 0) start = Math.max(0, t - 0.5 / cs.length);
          buf += ch; conf += c; cnt++;
        });
        flush(1);
      }
      return words;
    },
  };
}
