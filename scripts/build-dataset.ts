// Build training/test feature matrices for the two classifiers.
//   gt  : FUNSD ground-truth entities (the standard benchmark setting)
//   ocr : entities produced by our own Tesseract + grouping pipeline
// usage: npx tsx scripts/build-dataset.ts
import fs from 'node:fs';
import path from 'node:path';
import { LABELS, type Entity } from '../src/core/types';
import { candidatePairs, entityFeatures, pageCtx, pairFeatures } from '../src/core/layout';
import { groupWords } from '../src/core/layout';
import { DATA, gtAsEntities, loadGt, loadOcr, matchToGt, pages } from './funsd';

type Rows = { X: number[][]; y: number[]; g: number[] };
const empty = (): Rows => ({ X: [], y: [], g: [] });

function addPage(ents: Entity[], labels: (string | null)[], gtIds: (number | null)[], qa: Set<string>, W: number, H: number, pi: number, E: Rows, P: Rows) {
  const ctx = pageCtx(ents, W, H);
  const F = entityFeatures(ents, ctx);
  ents.forEach((_, i) => {
    E.X.push(F[i]);
    E.y.push(LABELS.indexOf((labels[i] ?? 'other') as any));
    E.g.push(pi);
  });
  const qIdx = ents.map((_, i) => i).filter((i) => labels[i] === 'question');
  const aIdx = ents.map((_, i) => i).filter((i) => labels[i] === 'answer');
  const pairs = candidatePairs(ents, qIdx, aIdx, ctx);
  const PF = pairFeatures(ents, pairs, ctx);
  pairs.forEach((p, k) => {
    P.X.push(PF[k]);
    P.y.push(qa.has(`${gtIds[p.q]},${gtIds[p.a]}`) ? 1 : 0);
    P.g.push(pi);
  });
}

const out: Record<string, { ent: Rows; pair: Rows }> = {};
let pageId = 0;
for (const split of ['training', 'testing'] as const) {
  for (const src of ['gt', 'ocr'] as const) {
    const E = empty(), P = empty();
    let missed = 0;
    for (const f of pages(split)) {
      const ocr = loadOcr(split, 'full', f);
      if (!ocr) { missed++; continue; }
      const gt = loadGt(split, f, ocr);
      pageId++;
      if (src === 'gt') {
        const ents = gtAsEntities(gt);
        addPage(ents, ents.map((e) => e.label!), ents.map((e) => e.id), gt.qa, ocr.width, ocr.height, pageId, E, P);
      } else {
        // text blocks as produced by both readers (Tesseract and PP-OCRv5), so one model serves either
        for (const mode of ['full', 'pp']) {
          const o = mode === 'full' ? ocr : loadOcr(split, mode, f);
          if (!o) continue;
          const ents = groupWords(o.words);
          const m = matchToGt(ents, gt);
          addPage(ents, m.map((g) => g?.label ?? 'other'), m.map((g) => g?.id ?? null), gt.qa, ocr.width, ocr.height, pageId, E, P);
        }
      }
    }
    out[`${split}-${src}`] = { ent: E, pair: P };
    console.log(split, src, 'entities', E.X.length, 'pairs', P.X.length, 'pos', P.y.filter(Boolean).length, missed ? `(missing ocr ${missed})` : '');
  }
}
fs.mkdirSync(path.join(DATA, 'features'), { recursive: true });
fs.writeFileSync(path.join(DATA, 'features', 'features.json'), JSON.stringify(out));
