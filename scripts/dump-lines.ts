// Debug aid for the made-up specimens only (never the private real documents):
// prints the text rows of specimens whose named fact is missing or wrong.
// usage: npx tsx scripts/dump-lines.ts <split> <type> <fact> [max=4]
import fs from 'node:fs';
import path from 'node:path';
import { classifyDocument, linesOf } from '../src/core/classify';
import { extractFacts } from '../src/core/facts';
import { DATA } from './funsd';

const [split = 'train', type = 'id_card', fact = 'Full name', max = '4'] = process.argv.slice(2);
const docs = JSON.parse(fs.readFileSync(path.join(DATA, 'synth', split, 'truth.json'), 'utf8')).documents as any[];
const nrm = (v: string) => v.toUpperCase().replace(/[^A-Z0-9]/g, '');
let n = 0;
for (const d of docs) {
  if (d.type !== type) continue;
  const key = Object.keys(d.fields).find((k) => (fact === 'number' ? /number/i.test(k) : k === fact));
  if (!key) continue;
  const c = path.join(DATA, 'synth-cache', split, d.file.replace(/\.[^.]+$/, '.json'));
  if (!fs.existsSync(c)) continue;
  const j = JSON.parse(fs.readFileSync(c, 'utf8'));
  const cls = classifyDocument(j.words, j.width, j.height, j.mrzFull ?? null);
  const facts = extractFacts(cls.type, j.words, [], j.mrzFull ?? null);
  const got = facts.find((x) => x.key === key) ?? (fact === 'number' ? facts.find((x) => /number/i.test(x.key)) : undefined);
  if (got && nrm(got.value) === nrm(d.fields[key])) continue;
  console.log(`\n=== ${d.file} sorted as ${cls.type} | ${key}: want "${d.fields[key]}" got "${got?.value ?? '-'}" | mrz ${j.mrzFull ? (j.mrzFull.valid ? 'ok' : 'bad') : 'none'}`);
  console.log(linesOf(j.words).map((l) => '   ' + l).join('\n'));
  if (j.mrzFull) console.log('   MRZ ' + j.mrzFull.lines.join(' / '));
  if (++n >= +max) break;
}
