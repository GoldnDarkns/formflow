// FUNSD loading helpers shared by the dataset builder and the evaluator.
import fs from 'node:fs';
import path from 'node:path';
import type { Box, Entity, Label, OcrWord } from '../src/core/types';
import { coverage, vOverlap, cy } from '../src/core/layout';
import { cleanWords } from '../src/core/ocr';

export const DATA = path.resolve(import.meta.dirname, '../../data');

export interface GtEntity { id: number; label: Label; text: string; box: Box; words: { text: string; box: Box }[]; linking: [number, number][] }
export interface GtPage { file: string; width: number; height: number; ents: GtEntity[]; qa: Set<string> }

export function pages(split: 'training' | 'testing'): string[] {
  return fs.readdirSync(path.join(DATA, 'funsd/dataset', split + '_data', 'annotations')).filter((f) => f.endsWith('.json')).sort();
}

export function loadGt(split: string, file: string, size?: { width: number; height: number }): GtPage {
  const j = JSON.parse(fs.readFileSync(path.join(DATA, 'funsd/dataset', split + '_data', 'annotations', file), 'utf8'));
  const ents: GtEntity[] = j.form.map((e: any) => ({ ...e, words: e.words.filter((w: any) => w.text.trim()) }));
  const byId = new Map(ents.map((e) => [e.id, e]));
  const qa = new Set<string>();
  for (const e of ents)
    for (const [a, b] of e.linking) {
      const A = byId.get(a), B = byId.get(b);
      if (!A || !B) continue;
      if (A.label === 'question' && B.label === 'answer') qa.add(`${a},${b}`);
      if (B.label === 'question' && A.label === 'answer') qa.add(`${b},${a}`);
    }
  return { file, width: size?.width ?? 0, height: size?.height ?? 0, ents, qa };
}

/** GT entities in the app's Entity format (word lines estimated by vertical overlap). */
export function gtAsEntities(p: GtPage): Entity[] {
  let line = 0;
  return p.ents
    .filter((e) => e.words.length && e.text.trim())
    .map((e) => {
      const ws = [...e.words].sort((a, b) => cy(a.box) - cy(b.box));
      const words: OcrWord[] = [];
      let last: Box | null = null;
      for (const w of ws) {
        if (last && vOverlap(last, w.box) < 0.4) line++;
        words.push({ text: w.text, box: w.box, conf: 100, line });
        last = w.box;
      }
      line++;
      return { id: e.id, text: e.text, box: e.box, words, label: e.label };
    });
}

export function loadOcr(split: string, mode: string, file: string): { width: number; height: number; words: OcrWord[]; ms: number; skew: number } | null {
  const f = path.join(DATA, 'ocr', `${split}-${mode}`, file);
  if (!fs.existsSync(f)) return null;
  const j = JSON.parse(fs.readFileSync(f, 'utf8'));
  return { ...j, words: cleanWords(j.words) };
}

/** map each OCR-derived entity to the GT entity most of its words fall inside */
export function matchToGt(ents: Entity[], p: GtPage): (GtEntity | null)[] {
  const gtWords = p.ents.flatMap((e) => e.words.map((w) => ({ e, box: w.box })));
  return ents.map((e) => {
    const votes = new Map<GtEntity, number>();
    for (const w of e.words) {
      let best: GtEntity | null = null, bc = 0.3;
      for (const g of gtWords) {
        const c = coverage(w.box, g.box);
        if (c > bc) { bc = c; best = g.e; }
      }
      if (!best) for (const g of p.ents) if (coverage(w.box, g.box) > 0.5) best = g;
      if (best) votes.set(best, (votes.get(best) ?? 0) + 1);
    }
    let top: GtEntity | null = null, n = 0;
    for (const [g, c] of votes) if (c > n) { n = c; top = g; }
    return top && n >= e.words.length / 2 ? top : null;
  });
}
