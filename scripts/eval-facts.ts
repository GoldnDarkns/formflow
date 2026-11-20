// Fast re-scoring of key facts from the private reading cache (no OCR): used while
// improving src/core/facts.ts. Specimens only, because only they come with the truth.
// usage: npx tsx scripts/eval-facts.ts [test|train] [synth]
import fs from 'node:fs';
import path from 'node:path';
import * as ortNode from 'onnxruntime-node';
import { classifyDocument, ID_TYPES } from '../src/core/classify';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { extractFacts } from '../src/core/facts';
import { DATA } from './funsd';

const split = process.argv[2] ?? 'test', root = process.argv[3] ?? 'synth';
const ROOT = path.resolve(import.meta.dirname, '..');
const clf = await loadClassifiers(ortNode as any, path.join(ROOT, 'public/models/entity_gbdt.onnx'), path.join(ROOT, 'public/models/link_gbdt.onnx'));
const docs = JSON.parse(fs.readFileSync(path.join(DATA, root, split, 'truth.json'), 'utf8')).documents as any[];
const nrm = (v: string) => v.toUpperCase().replace(/[^A-Z0-9]/g, '');
type Hit = 'right' | 'wrong' | 'missing' | 'verified-wrong';
const hits: { type: string; key: string; hit: Hit; trueType: boolean }[] = [];
for (const d of docs) {
  const c = path.join(DATA, `${root}-cache`, split, d.file.replace(/\.[^.]+$/, '.json'));
  if (!fs.existsSync(c)) continue;
  const j = JSON.parse(fs.readFileSync(c, 'utf8'));
  const mrz = j.mrzFull ?? null;
  const cls = classifyDocument(j.words, j.width, j.height, mrz);
  const an = await analyse(j.words, j.width, j.height, clf, undefined, ID_TYPES.includes(cls.type) ? 'id' : 'form', 'tesseract');
  const facts = extractFacts(cls.type, j.words, an.fields, mrz, process.env.SECOND === '0' ? [] : j.second ?? [], (process.env.PREFER as any) ?? 'second');
  for (const [k, v] of Object.entries(d.fields as Record<string, string>)) {
    const got = facts.find((x) => x.key === k) ?? (/^(Passport|Document) number$/.test(k) ? facts.find((x) => /^(Passport|Document) number$/.test(x.key)) : undefined);
    if (process.env.DEBUG && (!got || nrm(got.value) !== nrm(v))) console.log(`${d.type}|${cls.type}|${k}|want=${v}|got=${got?.value ?? '-'}|src=${got?.proof ?? ''}${got?.valid ? ' VALID' : ''}|mrz=${mrz ? (mrz.valid ? 'ok' : 'bad') : 'none'}`);
    hits.push({ type: d.type, key: /number/i.test(k) ? 'number' : k, hit: !got ? 'missing' : nrm(got.value) === nrm(v) ? 'right' : got.valid ? 'verified-wrong' : 'wrong', trueType: cls.type === d.type });
  }
}
const row = (label: string, hs: typeof hits) => { const c = (v: Hit) => hs.filter((h) => h.hit === v).length; return label.padEnd(20) + String(hs.length).padStart(5) + `${c('right')} (${Math.round((100 * c('right')) / Math.max(1, hs.length))}%)`.padStart(12) + String(c('wrong')).padStart(7) + String(c('missing')).padStart(9) + String(c('verified-wrong')).padStart(10); };
console.log(`${root}/${split}: key facts on ${docs.length} specimens`);
console.log('fact'.padEnd(20) + 'n'.padStart(5) + 'right'.padStart(12) + 'wrong'.padStart(7) + 'missing'.padStart(9) + ' verified-but-wrong');
for (const k of ['number', 'Full name', 'Date of birth', 'Expiry date']) console.log(row(k, hits.filter((h) => h.key === k)));
console.log(row('ALL', hits));
for (const t of [...new Set(hits.map((h) => h.type))]) console.log(row('  ' + t, hits.filter((h) => h.type === t)));
const out = { generated: new Date().toISOString(), specimens: docs.length, all: { n: hits.length, right: hits.filter((h) => h.hit === 'right').length, verifiedButWrong: hits.filter((h) => h.hit === 'verified-wrong').length }, byType: Object.fromEntries([...new Set(hits.map((h) => h.type))].map((t) => [t, { n: hits.filter((h) => h.type === t).length, right: hits.filter((h) => h.type === t && h.hit === 'right').length }])), byFact: Object.fromEntries(['number', 'Full name', 'Date of birth', 'Expiry date'].map((k) => [k, { n: hits.filter((h) => h.key === k).length, right: hits.filter((h) => h.key === k && h.hit === 'right').length }])) };
fs.writeFileSync(path.join(DATA, `${root}-facts-${split}.json`), JSON.stringify(out, null, 1));
