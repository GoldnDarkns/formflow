// Template mode experiment on the 20-form test set.
//
// Teach: one template per form from its BLANK page (../data/forms20/blank-img, made by
// make-forms20.py) plus the labelled boxes a person would draw (taken from truth.json).
// Test: the 20 filled-in copies, half of them degraded to look like a scan. Every page is
// matched against all 20 templates, so picking the right template is part of the test.
//
// Writes public/templates/library.json (the built-in template library the app ships)
// and the `templates` block of public/eval/results.json.
// usage: npx tsx scripts/eval-templates.ts
import fs from 'node:fs';
import path from 'node:path';
import type { Box } from '../src/core/types';
import { buildTemplate, matchTemplate, readTemplate, type Slot, type Template } from '../src/core/template';
import { reliability } from '../src/core/calibration';
import { sim } from './match';
import { DATA } from './funsd';
import { ocrAll, ROOT, type Reader } from './forms20-ocr';

const SET = path.join(ROOT, 'testsets/forms20');
const truth = JSON.parse(fs.readFileSync(path.join(SET, 'truth.json'), 'utf8'));
const files: string[] = truth.forms.map((f: any) => f.file);
const NAMES: Record<string, string> = {
  f14039: 'IRS Form 14039, Identity Theft Affidavit', f2553: 'IRS Form 2553, Election by a Small Business Corporation', f2848: 'IRS Form 2848, Power of Attorney',
  f3911: 'IRS Form 3911, Taxpayer Statement Regarding Refund', f4506: 'IRS Form 4506, Request for Copy of Tax Return', f4506t: 'IRS Form 4506-T, Request for Transcript',
  f4868: 'IRS Form 4868, Extension of Time to File', f56: 'IRS Form 56, Notice Concerning Fiduciary Relationship', f8332: 'IRS Form 8332, Release of Claim to Exemption',
  f8821: 'IRS Form 8821, Tax Information Authorization', f8822: 'IRS Form 8822, Change of Address', f8822b: 'IRS Form 8822-B, Change of Address (Business)',
  f8888: 'IRS Form 8888, Allocation of Refund', f9465: 'IRS Form 9465, Installment Agreement Request', fss4: 'IRS Form SS-4, Application for EIN',
  fw4: 'IRS Form W-4, Withholding Certificate', fw7: 'IRS Form W-7, Application for ITIN', fw8ben: 'IRS Form W-8BEN', fw9: 'IRS Form W-9, Request for Taxpayer ID',
  sf1152: 'OPM SF 1152, Designation of Beneficiary',
};

const R: any = { generated: new Date().toISOString(), protocol: 'Templates taught from the blank form (printed words as anchors, one labelled box per entry field); tested on the filled copies, each matched against all 20 templates', readers: {} };
let library: Template[] = [];

for (const reader of ['tesseract', 'ppocr'] as Reader[]) {
  const blank = await ocrAll(reader, path.join(DATA, 'forms20/blank-img'), files, 'blank');
  const filled = await ocrAll(reader, SET, files, 'ocr');
  const templates: Template[] = truth.forms.map((f: any) => {
    const b = blank.get(f.file)!;
    const slots: Slot[] = f.fields.map((x: any, i: number) => ({ label: (x.label || x.tooltip || `Box ${i + 1}`).trim(), box: [x.box[0] / f.width, x.box[1] / f.height, x.box[2] / f.width, x.box[3] / f.height] as Box }));
    const id = f.file.replace(/\.\w+$/, '');
    return buildTemplate(id, NAMES[id] ?? id, b.words, b.width, b.height, slots, f.source);
  });
  if (reader === 'tesseract') library = templates; // the app's default reader

  const tot: Record<string, { forms: number; matched: number; right: number; values: number; ok: number; exact: number; ms: number }> = {};
  const add = (k: string) => (tot[k] ??= { forms: 0, matched: 0, right: 0, values: 0, ok: 0, exact: 0, ms: 0 });
  const conf: { conf: number; ok: boolean }[] = [];
  const perForm: any[] = [];
  const misses: any[] = [];
  for (const f of truth.forms) {
    const o = filled.get(f.file)!;
    const t0 = performance.now();
    const m = matchTemplate(templates, o.words, o.width, o.height);
    const read = m ? readTemplate(m, o.words, o.width, o.height) : null;
    const ms = performance.now() - t0;
    const right = !!m && m.template.id === f.file.replace(/\.\w+$/, '');
    let ok = 0, exact = 0;
    f.fields.forEach((g: any, i: number) => {
      const got = right ? read!.fields[i] : undefined;
      const good = !!got && sim(got.value, g.value) >= 0.8;
      if (good) ok++;
      if (got && got.value.replace(/\s+/g, ' ').trim() === g.value.trim()) exact++;
      if (got?.value) conf.push({ conf: got.confidence, ok: good });
      if (!good && misses.length < 16) misses.push({ file: f.file, label: g.label.slice(0, 50), truth: g.value, got: got?.value ?? '(no template)' });
    });
    for (const k of ['all', f.quality]) { const t = add(k); t.forms++; t.matched += m ? 1 : 0; t.right += right ? 1 : 0; t.values += f.fields.length; t.ok += ok; t.exact += exact; t.ms += ms; }
    perForm.push({ file: f.file, quality: f.quality, matched: m?.template.id ?? null, score: m?.score ?? 0, values: f.fields.length, ok });
  }
  const rate = (t: (typeof tot)[string]) => ({ forms: t.forms, templateFound: t.matched, templateRight: t.right, typedValues: t.values, labelled: t.ok / t.values, exact: t.exact / t.values, msPerPage: t.ms / t.forms });
  const rel = reliability(conf, 5);
  R.readers[reader] = { all: rate(tot.all), clean: rate(tot.clean), scanLike: rate(tot['scan-like']), confidence: { fields: conf.length, meanShown: conf.reduce((a, x) => a + x.conf, 0) / Math.max(1, conf.length), shareRight: conf.filter((x) => x.ok).length / Math.max(1, conf.length), ece: rel.ece, bins: rel.bins }, perForm, examples: misses };
  const pc = (v: number) => (v * 100).toFixed(1) + '%';
  for (const [k, t] of Object.entries(tot)) console.log(`${reader.padEnd(9)} ${k.padEnd(9)} template right ${t.right}/${t.forms}  values ${t.values}  right label+value ${pc(t.ok / t.values)}  exact ${pc(t.exact / t.values)}  ${(t.ms / t.forms).toFixed(1)} ms/page`);
  console.log(`${reader.padEnd(9)} confidence: shown ${pc(R.readers[reader].confidence.meanShown)}, right ${pc(R.readers[reader].confidence.shareRight)}, ECE ${rel.ece.toFixed(3)}`);
  for (const p of perForm.filter((p) => p.ok / p.values < 0.6)) console.log('   weak:', p.file, p.quality, 'matched', p.matched, p.score.toFixed(2), `${p.ok}/${p.values}`);
}

fs.mkdirSync(path.join(ROOT, 'public/templates'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'public/templates/library.json'), JSON.stringify({ about: 'Built-in templates taught from blank public US federal forms (no personal data).', templates: library }));
const resPath = path.join(ROOT, 'public/eval/results.json');
const all = JSON.parse(fs.readFileSync(resPath, 'utf8'));
all.templates = R;
fs.writeFileSync(resPath, JSON.stringify(all, null, 1));
console.log('wrote public/templates/library.json and the templates block of public/eval/results.json');
