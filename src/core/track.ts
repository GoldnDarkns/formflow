// The live camera's document tracking: which finder to believe on a frame, a smoother for the
// outline between frames, and when a frame is good enough to take the picture by itself.
// Pure functions, shared by the camera (ui/Camera.tsx) and the test (scripts/eval-corners.ts).
import type { Pt, Quad } from './scan';

/** What each finder said about one frame. Corners are fractions (0..1) of the frame. */
export interface Finds {
  outline: Quad | null; // detectObject: a clean outline against a plain surface (exact, rarely wrong)
  edges: Quad | null; // detectCard: four straight edges (a card held in a hand), but also fooled by stripes and form boxes
  page: Quad | null; // detectPage: a bright page on a grainy surface
  net: { quad: Quad; score: number; plausible: boolean; contrast: number } | null; // the learned corner finder
}
export type How = 'outline' | 'network' | 'edges' | 'page';

// Thresholds chosen with scripts/eval-corners.ts on made-up photos (plain, patterned and hand-held
// backgrounds, photos with no card, flat form scans).
export const NET_SCORE = 0.5; // the network's "a document is here" probability
export const NET_CONTRAST = 0.45; // share of the outline with a clearly different surface outside it
export const NET_SOME = 0.1; // below this the network sees no document, so the edge finder is not believed either
const AGREE = 0.08; // corners this close (fraction of the frame) count as the same outline

const far = (a: Quad, b: Quad) => Math.max(...a.map((p) => Math.min(...b.map((c) => Math.hypot(p[0] - c[0], p[1] - c[1])))));
export const netSees = (f: Finds) => !!f.net && f.net.score >= NET_SCORE && f.net.plausible && f.net.contrast >= NET_CONTRAST;

/**
 * The outline to show. The classical outline finder first (exact on a plain surface, and it never fired
 * on an empty photo in the tests). Otherwise the network, using the edge finder's corners when the two
 * agree. The edge finder alone only when the network thinks a document may be there at all.
 * `strong` says whether the evidence is good enough to take the picture automatically.
 */
export function chooseOutline(f: Finds): { q: Quad; how: How; strong: boolean } | null {
  if (f.outline) return { q: f.outline, how: 'outline', strong: true };
  if (netSees(f)) {
    const n = f.net!;
    return f.edges && far(f.edges, n.quad) < AGREE ? { q: f.edges, how: 'edges', strong: true } : { q: n.quad, how: 'network', strong: true };
  }
  const score = f.net?.score ?? 1; // without the network (it failed to load) the classical finders are used as before
  if (f.edges && score >= NET_SOME) return { q: f.edges, how: 'edges', strong: false };
  if (f.page && score >= NET_SCORE) return { q: f.page, how: 'page', strong: false };
  return null;
}

/**
 * Smooths the outline between looks: each corner coordinate is tracked with its position and speed (an
 * alpha-beta filter, the simplest form of a Kalman filter). A measurement pulls the estimate part of the
 * way; between looks the outline keeps moving at its estimated speed, so it can be drawn every screen
 * frame instead of jumping ten times a second.
 */
export class QuadFilter {
  private x: number[] | null = null;
  private v = new Array(8).fill(0);
  private t = 0;
  private alpha: number;
  private beta: number;
  constructor(alpha = 0.55, beta = 0.12) { this.alpha = alpha; this.beta = beta; }
  reset() { this.x = null; this.v.fill(0); }
  get quad(): Quad | null { return this.x ? toQuad(this.x) : null; }
  update(q: Quad, now: number) {
    const z = q.flat();
    if (!this.x) { this.x = z; this.v.fill(0); this.t = now; return; }
    const dt = Math.max(1, now - this.t);
    this.t = now;
    for (let i = 0; i < 8; i++) {
      const pred = this.x[i] + this.v[i] * dt, r = z[i] - pred;
      this.x[i] = pred + this.alpha * r;
      this.v[i] = (this.v[i] + (this.beta * r) / dt) * 0.9; // a little drag, so the outline settles when the card stops
    }
  }
  /** where the outline is expected to be at `now` (for drawing between looks), never further than 150 ms ahead */
  at(now: number): Quad | null {
    if (!this.x) return null;
    const dt = Math.min(150, Math.max(0, now - this.t));
    return toQuad(this.x.map((c, i) => c + this.v[i] * dt));
  }
}
const toQuad = (a: number[]) => [[a[0], a[1]], [a[2], a[3]], [a[4], a[5]], [a[6], a[7]]] as Pt[] as Quad;

export const quadArea = (q: Quad) => Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * p[1], 0)) / 2;

export interface FrameQuality { sharpness: number; glary: boolean; blurry: boolean; paper: number }
/** What, if anything, stops a good picture right now (shown as the hint, and holds back auto-capture). */
export function frameProblem(q: Quad, quality: FrameQuality | null, sharpest: number): string | null {
  if (quadArea(q) < 0.06) return 'Move closer: the document is small in the picture.';
  if (!quality) return null;
  if (quality.paper < 70) return 'Too dark. Turn on more light.';
  if (quality.glary) return 'Glare on the document. Tilt it a little away from the light.';
  // blurred, or much less sharp than a moment ago: the hand or the card is still moving
  if (quality.blurry || quality.sharpness < sharpest * 0.8) return 'Hold steady: the picture is blurred.';
  return null;
}
