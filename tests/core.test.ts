// Unit tests for the pure core (no browser needed). Run: npm test
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { checkDigit, parseMrz, findMrzLines } from '../src/core/mrz';
import { parseDate, luhnOk } from '../src/core/facts';
import { levenshtein, similarity, substringDistance, scoreOcr } from '../src/core/ocr';
import { classifyDocument } from '../src/core/classify';
import { findPhotoArea } from '../src/core/photo';
import { analyse } from '../src/core/pipeline';
import { loadClassifiers } from '../src/core/classifier';
import { calibrate, CALIBRATION } from '../src/core/calibration';
import type { Box, GrayImage, OcrWord } from '../src/core/types';

const PASSPORT = ['P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<', 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'];
const field = (r: NonNullable<ReturnType<typeof parseMrz>>, k: string) => r.fields.find((f) => f.key === k)?.value;

test('MRZ check digit follows ICAO 9303 (7-3-1 weights)', () => {
  assert.equal(checkDigit('L898902C3'), 6);
  assert.equal(checkDigit('740812'), 2);
  assert.equal(checkDigit('120415'), 9);
});

test('MRZ: the ICAO specimen passport parses and every check passes', () => {
  const r = parseMrz(PASSPORT)!;
  assert.equal(r.format, 'TD3');
  assert.equal(r.valid, true);
  assert.equal(field(r, 'Document number'), 'L898902C3');
  assert.equal(field(r, 'Surname'), 'ERIKSSON');
  assert.equal(field(r, 'Given names'), 'ANNA MARIA');
});

test('MRZ: a misread digit in the date of birth is reported, not hidden', () => {
  const bad = [PASSPORT[0], PASSPORT[1].replace('7408122', '7409122')];
  const r = parseMrz(bad)!;
  assert.equal(r.valid, false);
  assert.equal(r.checks.find((c) => c.name === 'Date of birth')?.ok, false);
});

test('MRZ: an O read instead of 0 in the number is repaired by the check digits', () => {
  const r = parseMrz([PASSPORT[0], PASSPORT[1].replace('L898902C3', 'L8989O2C3')])!;
  assert.equal(field(r, 'Document number'), 'L898902C3');
  assert.equal(r.valid, true);
});

test('MRZ lines are found inside ordinary OCR text', () => {
  const lines = findMrzLines(['PASSPORT', 'Surname ERIKSSON', ...PASSPORT].join('\n'));
  assert.equal(lines.length, 2);
});

test('dates: day-first numbers, ISO and month names', () => {
  assert.equal(parseDate('12/08/1974'), '1974-08-12');
  assert.equal(parseDate('1974-08-12'), '1974-08-12');
  assert.equal(parseDate('12 AUG 1974'), '1974-08-12');
  assert.equal(parseDate('31/13/2020'), null);
  assert.equal(parseDate('no date here'), null);
});

test('Luhn check', () => {
  assert.equal(luhnOk('4539578763621486'), true);
  assert.equal(luhnOk('4539578763621487'), false);
});

test('text distances', () => {
  assert.equal(levenshtein('kitten', 'sitting'), 3);
  assert.equal(similarity('Anna Maria', 'ANNA  MARIA'), 1);
  assert.equal(substringDistance('maria', 'anna martia eriksson'), 1);
});

test('OCR scoring: a perfect read has zero error', () => {
  const gt = [{ text: 'Hello', box: [0, 0, 50, 10] as Box }, { text: 'world', box: [60, 0, 110, 10] as Box }];
  const s = scoreOcr(gt, gt);
  assert.equal(s.errors, 0);
  assert.equal(s.omitted, 0);
  assert.equal(s.hallucinated, 0);
});

const W = (text: string, x: number, y: number, h = 20, line = 0): OcrWord => ({ text, box: [x, y, x + text.length * h * 0.6, y + h], conf: 95, line });

test('sorter: a page with a valid passport MRZ is a passport', () => {
  const words = [W('PASSPORT', 40, 40), W(PASSPORT[0], 40, 600, 24, 1), W(PASSPORT[1], 40, 640, 24, 2)];
  const c = classifyDocument(words, 1250, 880, parseMrz(PASSPORT));
  assert.equal(c.type, 'passport');
  assert.ok(c.confidence > 0.9);
});

test('sorter: an Emirates ID number makes it an Emirates ID', () => {
  const words = [W('UNITED', 40, 40), W('ARAB', 200, 40), W('EMIRATES', 320, 40), W('ID', 40, 120, 20, 1), W('Number', 80, 120, 20, 1), W('784-1974-1234567-6', 40, 160, 24, 2)];
  assert.equal(classifyDocument(words, 1250, 790, null).type, 'emirates_id');
});

function page(w: number, h: number, bg = 232): GrayImage {
  return { width: w, height: h, data: new Uint8Array(w * h).fill(bg) };
}
function fillRect(g: GrayImage, b: Box, v: number) {
  for (let y = b[1]; y < b[3]; y++) for (let x = b[0]; x < b[2]; x++) g.data[y * g.width + x] = v;
}

test('photo finder: finds the portrait block on an ID card', () => {
  const g = page(640, 400);
  const photo: Box = [24, 60, 156, 236];
  fillRect(g, photo, 150);
  const text: Box[] = [[180, 70, 330, 90], [180, 100, 400, 124]];
  for (const t of text) fillRect(g, t, 20);
  const b = findPhotoArea(g, text)!;
  assert.ok(b, 'a photo area is found');
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(b[i] - photo[i]) <= 8, `edge ${i} within 8 px: ${b[i]} vs ${photo[i]}`);
});

test('photo finder: a page of text only has no photo', () => {
  const g = page(640, 800);
  const text: Box[] = [];
  for (let i = 0; i < 12; i++) { const t: Box = [40, 60 + i * 50, 400, 80 + i * 50]; text.push(t); fillRect(g, t, 20); }
  assert.equal(findPhotoArea(g, text), null);
});

test('photo finder: a wide banner is not mistaken for a portrait', () => {
  const g = page(640, 400);
  fillRect(g, [0, 0, 640, 50], 60);
  assert.equal(findPhotoArea(g, []), null);
});

test('calibration: stays in range, keeps the order, and lowers over-confident scores', () => {
  let prev = -1;
  for (let c = 0; c <= 1.0001; c += 0.05) {
    const v = calibrate(c, 'tesseract');
    assert.ok(v >= 0 && v <= 1);
    assert.ok(v >= prev - 1e-9, 'monotonic');
    prev = v;
  }
  assert.ok(CALIBRATION.fitted ? calibrate(0.95, 'tesseract') < 0.95 : true);
});

test('layout models: "Name: Anna Eriksson" becomes a field', async () => {
  const ort = await import('onnxruntime-node');
  const root = path.resolve(import.meta.dirname, '..');
  const clf = await loadClassifiers(ort as any, path.join(root, 'public/models/entity_gbdt.onnx'), path.join(root, 'public/models/link_gbdt.onnx'));
  const words = [
    W('APPLICATION', 300, 40, 28, 0), W('FORM', 520, 40, 28, 0),
    W('Name:', 60, 160, 20, 1), W('Anna', 220, 160, 20, 1), W('Eriksson', 290, 160, 20, 1),
    W('Date:', 60, 220, 20, 2), W('12/08/1974', 220, 220, 20, 2),
    W('City:', 60, 280, 20, 3), W('Dubai', 220, 280, 20, 3),
  ];
  const an = await analyse(words, 850, 1100, clf);
  const name = an.fields.find((f) => /name/i.test(f.key));
  assert.ok(name, 'a Name field exists');
  assert.equal(name!.value, 'Anna Eriksson');
  for (const f of an.fields) assert.ok(f.confidence >= 0 && f.confidence <= 1);
});

// ---------- template mode ----------
import { buildTemplate, matchTemplate, readTemplate } from '../src/core/template';
import { fitIsotonic, applyCurve, reliability } from '../src/core/calibration';

const PRINT = ['Request', 'for', 'Taxpayer', 'Identification', 'Number', 'and', 'Certification', 'Name', 'Business', 'Address', 'City', 'State', 'Signature', 'Date', 'Department', 'Treasury', 'Internal', 'Revenue', 'Service', 'Purpose', 'Instructions', 'Exemptions', 'Requester', 'Account'];
function blankPage(): OcrWord[] {
  // printed words on a 4-column grid of an 850 x 1100 page
  return PRINT.map((t, i) => W(t, 60 + (i % 4) * 190, 80 + Math.floor(i / 4) * 150, 18, i));
}
const SLOTS = [{ label: 'Name', box: [0.07, 0.10, 0.45, 0.13] as Box }, { label: 'City', box: [0.07, 0.24, 0.45, 0.27] as Box }, { label: 'Date', box: [0.55, 0.10, 0.9, 0.13] as Box }];
const moved = (ws: OcrWord[], k: number, dx: number, dy: number): OcrWord[] => ws.map((w) => ({ ...w, box: [w.box[0] * k + dx, w.box[1] * k + dy, w.box[2] * k + dx, w.box[3] * k + dy] as Box }));

test('template: a shifted, resized copy is recognised and its boxes are read', () => {
  const t = buildTemplate('t1', 'Test form', blankPage(), 850, 1100, SLOTS);
  const filled = [...blankPage(), W('Mira', 70, 118, 18, 90), W('Halvorsen', 130, 118, 18, 90), W('Fairhaven', 70, 272, 18, 91), W('03/14/2025', 480, 118, 18, 92)];
  const page = moved(filled, 1.04, 14, -9);
  const m = matchTemplate([t], page, 850, 1100)!;
  assert.ok(m, 'template matches');
  assert.ok(m.score > 0.9);
  const r = readTemplate(m, page, 850, 1100);
  assert.deepEqual(r.fields.map((f) => [f.key, f.value]), [['Name', 'Mira Halvorsen'], ['City', 'Fairhaven'], ['Date', '03/14/2025']]);
  assert.ok(r.fields.every((f) => f.source === 'template'));
});

test('template: an empty box is reported as empty, not invented', () => {
  const t = buildTemplate('t1', 'Test form', blankPage(), 850, 1100, SLOTS);
  const m = matchTemplate([t], blankPage(), 850, 1100)!;
  const r = readTemplate(m, blankPage(), 850, 1100);
  assert.ok(r.fields.every((f) => f.value === '' && f.status === 'empty'));
});

test('template: an unrelated page does not match', () => {
  const t = buildTemplate('t1', 'Test form', blankPage(), 850, 1100, SLOTS);
  const other = ['Invoice', 'Total', 'Quantity', 'Price', 'Customer', 'Order', 'Shipping', 'Thank', 'you', 'Payment', 'Terms', 'Net'].map((x, i) => W(x, 60 + (i % 3) * 200, 100 + Math.floor(i / 3) * 120, 18, i));
  assert.equal(matchTemplate([t], other, 850, 1100), null);
});

test('template: the right one is chosen among several', () => {
  const a = buildTemplate('a', 'Form A', blankPage(), 850, 1100, SLOTS);
  const bWords = blankPage().map((w, i) => ({ ...w, text: 'Other' + i }));
  const b = buildTemplate('b', 'Form B', bWords, 850, 1100, SLOTS);
  assert.equal(matchTemplate([b, a], moved(blankPage(), 0.98, 5, 5), 850, 1100)!.template.id, 'a');
});

test('isotonic fit: monotone, and brings an over-confident score back to the truth', () => {
  // a score that says 0.9 while only about half are right
  const s: { score: number; ok: boolean }[] = [];
  for (let i = 0; i < 400; i++) { const score = 0.5 + (i % 50) / 100; s.push({ score, ok: (i * 7919) % 100 < score * 55 }); }
  const c = fitIsotonic(s);
  for (let i = 1; i < c.y.length; i++) assert.ok(c.y[i] >= c.y[i - 1]);
  const before = reliability(s.map((x) => ({ conf: x.score, ok: x.ok }))).ece;
  const after = reliability(s.map((x) => ({ conf: applyCurve(c, x.score), ok: x.ok }))).ece;
  assert.ok(after < before / 3, `ECE ${before.toFixed(3)} -> ${after.toFixed(3)}`);
});

// ---------- checkboxes and signatures ----------
import { findCheckboxes, findSignatures, checkboxLabel } from '../src/core/marks';

function outline(g: GrayImage, b: Box, t = 2) {
  fillRect(g, [b[0], b[1], b[2], b[1] + t], 0); fillRect(g, [b[0], b[3] - t, b[2], b[3]], 0);
  fillRect(g, [b[0], b[1], b[0] + t, b[3]], 0); fillRect(g, [b[2] - t, b[1], b[2], b[3]], 0);
}
function cross(g: GrayImage, b: Box) {
  const n = b[2] - b[0];
  for (let i = 3; i < n - 3; i++) for (let k = 0; k < 2; k++) { g.data[(b[1] + i) * g.width + b[0] + i + k] = 0; g.data[(b[1] + i) * g.width + b[2] - 1 - i - k] = 0; }
}

test('checkboxes: outlines are found and a cross inside means ticked', () => {
  const g = page(1000, 600, 250);
  const empty: Box = [100, 100, 120, 120], ticked: Box = [300, 100, 320, 120];
  outline(g, empty); outline(g, ticked); cross(g, ticked);
  fillRect(g, [500, 100, 520, 120], 0); // a solid block is not a checkbox
  fillRect(g, [100, 300, 700, 303], 0); // nor is a ruled line
  const found = findCheckboxes(g);
  assert.equal(found.length, 2);
  assert.deepEqual(found.map((c) => c.checked), [false, true]);
  assert.ok(Math.abs(found[1].box[0] - 300) <= 2);
});

test('checkbox label: the words printed just right of the box, up to the next box', () => {
  const words = [W('Individual', 130, 100, 18, 0), W('C', 330, 100, 18, 0), W('corporation', 350, 100, 18, 0)];
  assert.equal(checkboxLabel([100, 100, 120, 120], words, [[300, 100, 320, 120]]), 'Individual');
  assert.equal(checkboxLabel([300, 100, 320, 120], words, [[100, 100, 120, 120]]), 'C corporation');
});

test('signatures: a pen stroke above the caption is "signed", a bare line is not', () => {
  const make = (signed: boolean) => {
    const g = page(1000, 400, 250);
    fillRect(g, [100, 200, 600, 202], 0); // the signing line
    if (signed) for (let x = 120; x < 420; x++) { const y = Math.round(180 + 14 * Math.sin(x / 9) + 5 * Math.sin(x / 3)); fillRect(g, [x, y, x + 2, y + 5], 0); }
    return g;
  };
  const caption = [{ text: 'Signature', box: [100, 208, 190, 224] as Box, conf: 95, line: 0 }, { text: 'Date', box: [640, 208, 680, 224] as Box, conf: 95, line: 0 }];
  assert.equal(findSignatures(make(true), caption)[0].signed, true);
  assert.equal(findSignatures(make(false), caption)[0].signed, false);
  assert.equal(findSignatures(make(true), [caption[1]]).length, 0, 'no caption, nothing reported');
});

// ---------- ZIP input ----------
import { zipSync, strToU8 } from 'fflate';
import { unzipScans } from '../src/ui/bundle';

test('ZIP input: images and PDFs come out in name order; folders, hidden files and other types are skipped', async () => {
  const z = zipSync({
    'pile/scan10.jpg': new Uint8Array([1, 2, 3]),
    'pile/scan2.PNG': new Uint8Array([4, 5]),
    'pile/form.pdf': new Uint8Array([6]),
    'pile/notes.txt': strToU8('not a scan'),
    '__MACOSX/pile/._scan2.PNG': new Uint8Array([9]),
    'pile/.hidden.jpg': new Uint8Array([9]),
  });
  const out = await unzipScans(new Blob([z as BlobPart]));
  assert.deepEqual(out.map((e) => e.name), ['form.pdf', 'scan2.PNG', 'scan10.jpg']);
  assert.deepEqual(out.map((e) => e.pdf), [true, false, false]);
  assert.equal(out[1].blob.type, 'image/png');
  assert.equal(out[2].blob.size, 3);
});


// ---------- lessons from real documents ----------
import { believableMrz, mrzCrops, readingQuality } from '../src/core/orient';
import { CLUES, priorModel, sorterFeatures, classifyFeatures } from '../src/core/classify';

test('a sideways page reads as almost no clean words', () => {
  const upright = ['Driving', 'Licence', 'Name', 'Nationality', 'Date', 'Birth'].map((t, i) => ({ ...W(t, 40, 40 + i * 30), conf: 92 }));
  const sideways = ['lI', 'rn', '1l|', 'VVe', 'ii'].map((t, i) => ({ ...W(t, 40, 40 + i * 30), conf: 35 }));
  assert.ok(readingQuality(upright) >= 4);
  assert.equal(readingQuality(sideways), 0);
});

test('code lines found only by searching the page bottom need two passing check digits', () => {
  const junk = { valid: false, checks: [{ ok: false }, { ok: true }, { ok: false }] };
  const good = { valid: false, checks: [{ ok: true }, { ok: true }, { ok: false }] };
  assert.equal(believableMrz(junk, false), false);
  assert.equal(believableMrz(good, false), true);
  assert.equal(believableMrz(junk, true), true, 'kept when the page really showed rows of <');
  assert.equal(believableMrz(null, true), false);
});

test('code-line search: long text pages with no "<" are skipped, others get four places twice', () => {
  assert.equal(mrzCrops(null, 2000, 400).length, 0);
  assert.equal(mrzCrops(null, 2000, 80).length, 4);
  assert.equal(mrzCrops([1500, 1700], 2000, 80).length, 8);
});

test('sorter: a vehicle licence is not a driving licence', () => {
  const words = ['UNITED', 'ARAB', 'EMIRATES', 'Vehicle', 'License', 'Traffic', 'Plate', 'No.', 'Place', 'of', 'Issue', 'Owner', 'Exp.', 'Date', 'Ins.', 'Exp.', 'Policy', 'No.', 'Mortgage', 'By'].map((t, i) => W(t, 40 + (i % 5) * 150, 40 + Math.floor(i / 5) * 40, 20, Math.floor(i / 5)));
  assert.equal(classifyFeatures(sorterFeatures(words, 1200, 750, null), priorModel()).type, 'vehicle_licence');
  assert.equal(classifyDocument(words, 1200, 750, null).type, 'vehicle_licence');
});

test('sorter: an international driving permit in French is a driving licence', () => {
  const words = ['PERMIS', 'INTERNATIONAL', 'DE', 'CONDUIRE', 'Convention', 'on', 'Road', 'Traffic', 'Valid', 'until'].map((t, i) => W(t, 40 + (i % 4) * 180, 40 + Math.floor(i / 4) * 40, 20, Math.floor(i / 4)));
  assert.equal(classifyDocument(words, 900, 1200, null).type, 'driving_licence');
});

test('sorter model: one number per clue and type, nothing else', () => {
  const m = JSON.parse(fs.readFileSync(path.resolve(import.meta.dirname, '../src/core/sorter.json'), 'utf8'));
  if (!m.fitted) return;
  assert.deepEqual(Object.keys(m).sort(), ['bias', 'clues', 'fitted', 'note', 'types', 'weights']);
  assert.deepEqual(m.clues, CLUES.map((c) => c.id));
  assert.ok(m.weights.every((r: unknown[]) => r.length === CLUES.length && r.every((v) => typeof v === 'number')));
});

test('MRZ: padding misread as K and L does not end up in the name', () => {
  const r = parseMrz([PASSPORT[0].replace(/<{19}$/, 'KLLLLLLLLLLLLKLLKLK'), PASSPORT[1]])!;
  assert.equal(field(r, 'Surname'), 'ERIKSSON');
  assert.equal(field(r, 'Given names'), 'ANNA MARIA');
});

// ---------- key facts on cards without code lines ----------
import { extractFacts } from '../src/core/facts';
const lineWords = (lines: string[]): OcrWord[] => lines.flatMap((l, i) => l.split(' ').map((t, k) => ({ text: t, box: [40 + k * 110, 40 + i * 40, 140 + k * 110, 62 + i * 40] as Box, conf: 92, line: i })));
const fact = (fs: ReturnType<typeof extractFacts>, k: string) => fs.find((f) => f.key === k)?.value;

test('key facts: an EU-style licence numbers its fields instead of labelling them', () => {
  const f = extractFacts('driving_licence', lineWords(['DRIVING LICENCE', '1. DEMIR', '2. PRIYA', '3. 17-02-1967 PORT ELIAS', '4a. 03-07-2019', '4b. 02-07-2027', '4c. TRANSPORT AUTHORITY', '5. 1300777', '9. AM B']), [], null);
  assert.equal(fact(f, 'Full name'), 'PRIYA DEMIR');
  assert.equal(fact(f, 'Licence number'), '1300777');
  assert.equal(fact(f, 'Date of birth'), '1967-02-17');
  assert.equal(fact(f, 'Expiry date'), '2027-07-02');
});

test('key facts: a table-style licence', () => {
  const f = extractFacts('driving_licence', lineWords(['Driving License', 'License No. 4055325', 'Name TOMAS NOVAK', 'Nationality SERBIAN', 'Date of Birth 10/11/2000', 'Issue Date 14/10/2020', 'Expiry Date 14/10/2027', 'Place of Issue Dubai']), [], null);
  assert.equal(fact(f, 'Licence number'), '4055325');
  assert.equal(fact(f, 'Full name'), 'TOMAS NOVAK');
  assert.equal(fact(f, 'Expiry date'), '2027-10-14');
});

test('key facts: a vehicle licence gives plate, owner, chassis and dates', () => {
  const f = extractFacts('vehicle_licence', lineWords(['Vehicle License', 'Traffic Plate No. AA / 73902', 'Owner SUMMIT RENT A CAR LLC', 'Exp. Date 2027-07-16 Reg. Date 2025-05-20', 'Ins. Exp. 2027-08-16', 'Chassis No. MDHBN7ADXNG137207']), [], null);
  assert.equal(fact(f, 'Plate number'), 'AA / 73902');
  assert.equal(fact(f, 'Owner'), 'SUMMIT RENT A CAR LLC');
  assert.equal(fact(f, 'Chassis number'), 'MDHBN7ADXNG137207');
  assert.equal(fact(f, 'Expiry date'), '2027-07-16');
  assert.equal(fact(f, 'Registration date'), '2025-05-20');
  assert.equal(fact(f, 'Insurance expiry'), '2027-08-16');
});

test('key facts: surname and given names printed separately are joined', () => {
  const f = extractFacts('id_card', lineWords(['NATIONAL IDENTITY CARD', 'Surname / Nom', 'SOKOLOV', 'Given names / Prénoms', 'AISHA', 'ID Number', '091201355', 'Date of birth', '14 DEC 1991']), [], null);
  assert.equal(fact(f, 'Full name'), 'AISHA SOKOLOV');
  assert.equal(fact(f, 'Document number'), '091201355');
  assert.equal(fact(f, 'Date of birth'), '1991-12-14');
});

// ---- second accuracy pass: names mended from the printed text, code-line shape, photo quality ----
import { mendName, editDistance, alertsFor } from '../src/core/facts';
import { shapedLikeMrz } from '../src/core/orient';
import { photoQuality } from '../src/core/quality';

test('a name from the code line is mended from the printed words, and confirmed when both agree', () => {
  const page = lineWords(['REPUBLIC OF UTOPIA', 'ALMEIDA YUKI', 'P<UTOALMEIDA<<YUKI<<<<<<<<']);
  assert.deepEqual(mendName('UKI ALMEIDA', page), { name: 'YUKI ALMEIDA', twice: false });
  assert.deepEqual(mendName('YUKI ALMEIDA', page), { name: 'YUKI ALMEIDA', twice: true });
  // a correct name is never swapped for a longer word that merely contains it
  assert.equal(mendName('ANA OLIVEIRA', lineWords(['ANA', 'P<NGAOLIVEIRA<<ANA<<<<'])).name, 'ANA OLIVEIRA');
  assert.equal(editDistance('sarmame', 'surname'), 2);
});

test('a final L or K of a name survives the padding clean-up', () => {
  const m = parseMrz(['P<SRBOKAFOR<<RAFAEL<<<<<<<<<<<<<<<<<<<<<<<<<', 'BV628593<1SRB6606194M3105157<<<<<<<<<<<<<<<0'])!;
  assert.equal(m.fields.find((f) => f.key === 'Given names')?.value, 'RAFAEL');
});

test('misread labels and side-by-side name columns still give the name', () => {
  assert.equal(fact(extractFacts('driving_licence', lineWords(['INTERNATIONAL DRIVING PERMIT', '1 Sarmame 0 IONESCU', '2. Olver names KOFI', '4, Date of beth 26/01/1968']), [], null), 'Full name'), 'KOFI IONESCU');
  const cols: OcrWord[] = [['Surname', 40, 40], ['Given', 400, 40], ['names', 470, 40], ['MENSAH', 40, 80], ['SOFIA', 400, 80], ['ELENA', 480, 80]].map(([t, x, y]) => ({ text: t as string, box: [x as number, y as number, (x as number) + 70, (y as number) + 22] as Box, conf: 92, line: 0 }));
  assert.equal(fact(extractFacts('id_card', cols, [], null), 'Full name'), 'SOFIA ELENA MENSAH');
});

test('ordinary text is not taken for code lines just because two check digits pass', () => {
  assert.equal(shapedLikeMrz(['EXPDATE20340302REGDATE20240303<<<<<<', 'INSEXP20340302INSURERNORTHSTARINSURA'], 'TD2'), false);
  assert.equal(shapedLikeMrz(['P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<', 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'], 'TD3'), true);
  assert.equal(believableMrz({ valid: false, checks: [{ ok: true }, { ok: true }, { ok: false }], lines: ['EXPDATE20340302REGDATE20240303<<<<<<', 'X'], format: 'TD2' }, false), false);
});

test('photo check: a blurred page scores lower than a sharp one, and a burnt-out patch counts as glare', () => {
  const W = 600, H = 400;
  const sharp: GrayImage = { width: W, height: H, data: new Uint8Array(W * H).fill(215) };
  for (let y = 40; y < H - 40; y += 24) for (let x = 40; x < W - 40; x += 9) if ((x * 7 + y) % 5) for (let yy = 0; yy < 12; yy++) for (let xx = 0; xx < 3; xx++) sharp.data[(y + yy) * W + x + xx] = 30;
  const blur: GrayImage = { width: W, height: H, data: new Uint8Array(sharp.data) };
  for (let pass = 0; pass < 6; pass++) { const src = new Uint8Array(blur.data); for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) { let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += src[(y + dy) * W + x + dx]; blur.data[y * W + x] = s / 9; } }
  const a = photoQuality(sharp), b = photoQuality(blur);
  assert.ok(a.sharpness > b.sharpness * 1.5 && !a.blurry && b.blurry, `${a.sharpness} vs ${b.sharpness}`);
  assert.equal(a.glary, false);
  const glare: GrayImage = { width: W, height: H, data: new Uint8Array(sharp.data) };
  for (let y = 100; y < 220; y++) for (let x = 200; x < 380; x++) glare.data[y * W + x] = 255;
  assert.equal(photoQuality(glare).glary, true);
  assert.match(alertsFor('passport', 0.99, [], null, new Date(), { blurry: true, glary: false })[0].text, /blurry/);
  assert.equal(alertsFor('form', 0.99, [], null, new Date(), { blurry: false, glary: true }).length, 0);
});

test('two readings of one card: agreement is marked, a gap is filled, and dates follow birth-first, expiry-last', () => {
  const first = lineWords(['DRIVING LICENCE', 'Surname BIANCHI', 'Given names MARCO', 'Date of birth 12.03.1990', 'Date of expiry 01.07.2029', 'Licence Mo. AB12345']);
  const other = lineWords(['DRIVING LICENCE', 'Surname BIANCHI', 'Given names MARCO', 'Date of birth 02.07.2019', 'Date of expiry 02.07.2019', 'Licence No. AB123456', 'Convention on Road Traffic of 8 November 1968']);
  const f = extractFacts('driving_licence', first, [], null, other);
  const get = (k: string) => f.find((x) => x.key === k);
  assert.equal(get('Full name')?.value, 'MARCO BIANCHI');
  assert.equal(get('Full name')?.twice, true);
  assert.equal(get('Licence number')?.value, 'AB123456'); // only the second reader got the label
  assert.equal(get('Date of birth')?.value, '1990-03-12'); // the earlier of two candidates
  assert.equal(get('Expiry date')?.value, '2029-07-01'); // the later of two candidates
  assert.equal(get('Expiry date')?.alt, '2019-07-02');
  // without a second reading nothing changes
  assert.equal(extractFacts('driving_licence', first, [], null).find((x) => x.key === 'Full name')?.twice, undefined);
});

test('a code line cannot confirm its own misreading of a name', () => {
  const page = lineWords(['PASSPORT', 'ALMEIDA', 'YUKI', 'KARIM', 'P<UTOALMEIDA<<UKI<KARIMK<<<<<<<<']);
  assert.deepEqual(mendName('UKI KARIMK ALMEIDA', page), { name: 'YUKI KARIM ALMEIDA', twice: false });
  // a label word is never taken for part of a name
  assert.equal(mendName('ANA DE SOUZA', lineWords(['DATA', 'DATE', 'SOUZA', 'I<UTO<<<', 'SOUZA<<ANA<DE<<<'])).name, 'ANA DE SOUZA');
});

import { detectPage } from '../src/core/scan';
import { binariseAuto, inkShare, SPECKLED } from '../src/core/preprocess';

test('page finder: a dark patch that eats one corner of the page does not move that corner', () => {
  const W = 600, H = 800, data = new Uint8Array(W * H).fill(60);
  // a light page from (100,120) to (500,700), with a dark photo in its bottom-left corner reaching the page edge
  for (let y = 120; y < 700; y++) for (let x = 100; x < 500; x++) data[y * W + x] = y > 520 && x < 230 ? 60 : 232;
  const q = detectPage({ width: W, height: H, data });
  assert.ok(q, 'page found');
  const want = [[100, 120], [500, 120], [500, 700], [100, 700]];
  q!.forEach((p, i) => assert.ok(Math.hypot(p[0] - want[i][0], p[1] - want[i][1]) < 14, `corner ${i} at ${p.map(Math.round)}`));
});

test('clean-up: a grainy surface is recognised as speckle, plain print is not', () => {
  const W = 400, H = 400;
  let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const grain = new Uint8Array(W * H), print = new Uint8Array(W * H).fill(235);
  for (let i = 0; i < grain.length; i++) grain[i] = 150 + Math.round((rnd() - 0.5) * 160 + (i % W) * 0.2);
  for (let y = 40; y < 360; y += 40) for (let x = 40; x < 360; x++) for (let k = 0; k < 4; k++) print[(y + k) * W + x] = 30;
  for (let i = 0; i < print.length; i++) print[i] = Math.max(0, print[i] - Math.round((i % W) * 0.12)); // uneven light
  assert.ok(inkShare(binariseAuto({ width: W, height: H, data: grain }).img) > SPECKLED);
  assert.ok(inkShare(binariseAuto({ width: W, height: H, data: print }).img) < SPECKLED);
});

import { detectObject } from '../src/core/scan';
import { tidyIdFields } from '../src/core/facts';

// a flat-coloured picture with a tilted four-sided patch painted on it
function paint(W: number, H: number, bg: number[], card: number[] | null, extra?: (x: number, y: number) => number[] | null): Uint8ClampedArray {
  const px = new Uint8ClampedArray(W * H * 4);
  const a = 0.14, cx = W / 2, cy = H / 2, hw = W * 0.32, hh = H * 0.2;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const dx = x - cx, dy = y - cy, u = dx * Math.cos(a) + dy * Math.sin(a), v = -dx * Math.sin(a) + dy * Math.cos(a);
      const c = extra?.(x, y) ?? (card && Math.abs(u) < hw && Math.abs(v) < hh ? card : bg);
      px.set([c[0], c[1], c[2], 255], (y * W + x) * 4);
    }
  return px;
}

test('outline finder: a pale card on a white table is found, where the bright-page finder sees nothing', () => {
  const W = 600, H = 800, px = paint(W, H, [250, 250, 248], [226, 232, 214]);
  const g: GrayImage = { width: W, height: H, data: new Uint8Array(W * H).map((_, i) => px[i * 4 + 1]) };
  assert.equal(detectPage(g), null);
  const q = detectObject(px, W, H);
  assert.ok(q, 'the card was not found');
  // corners within 3% of the diagonal of where they were painted
  const a = 0.14, hw = W * 0.32, hh = H * 0.2;
  const truth = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([u, v]) => [W / 2 + u * Math.cos(a) - v * Math.sin(a), H / 2 + u * Math.sin(a) + v * Math.cos(a)]);
  for (const t of truth) assert.ok(Math.min(...q!.map((p) => Math.hypot(p[0] - t[0], p[1] - t[1]))) < Math.hypot(W, H) * 0.03);
});

test('outline finder: a dark card on a light desk is found too, and an empty table gives nothing', () => {
  assert.ok(detectObject(paint(600, 800, [214, 208, 196], [40, 52, 96]), 600, 800));
  assert.equal(detectObject(paint(600, 800, [214, 208, 196], null), 600, 800), null);
});

test('outline finder: a box printed on a scanned page is not cut out', () => {
  // white page, a boxed panel of the same white inside it, and lines of "text" all over the page
  const W = 600, H = 800;
  const px = paint(W, H, [255, 255, 255], null, (x, y) => {
    const box = x > 120 && x < 480 && y > 250 && y < 520;
    if (box && (x < 123 || x > 477 || y < 253 || y > 517)) return [0, 0, 0];
    if (y % 22 < 5 && x > 40 && x < 560 && (x >> 3) % 3) return [20, 20, 20];
    return null;
  });
  assert.equal(detectObject(px, W, H), null);
});

test('identity documents: noise pairs are dropped, real printed labels are kept and spelled properly', () => {
  const f = (key: string, value: string, confidence = 0.8) => ({ key, value, confidence, status: 'ok' as const, reasons: [], source: 'layout' as const });
  const out = tidyIdFields([f('FUND! A! GOGETT 92', 'o ag', 0.42), f('HIJ', 'J!', 0.69), f('LICEN®', 'LL', 0.54), f('ATIONALIT', 'UTOPIA'), f('Place of Issue', 'Port Sample'), f('Date of Birth', '1990-01-01')], [{ key: 'Date of birth', value: '1990-01-01', from: 'text' } as any]);
  assert.deepEqual(out.map((x) => [x.key, x.value]), [['Nationality', 'UTOPIA'], ['Place of issue', 'Port Sample']]);
});

import { detectCard, quadHoldsWords } from '../src/core/scan';

test('four-edges finder: a card over a busy background with a finger across one edge is found', () => {
  const W = 640, H = 480;
  // background of coloured rings, a pale card, and a skin-coloured blob over its right edge
  const card = (x: number, y: number) => x > 200 && x < 470 && y > 150 && y < 320;
  const px = paint(W, H, [0, 0, 0], null, (x, y) => {
    if ((x - 475) ** 2 / 45 ** 2 + (y - 230) ** 2 / 18 ** 2 < 1) return [214, 168, 140];
    if (card(x, y)) return (y % 30 < 3 && x > 230 && x < 400) ? [60, 60, 70] : [232, 236, 226];
    return [[90, 120, 200], [200, 90, 80], [70, 160, 110], [180, 170, 60]][(Math.hypot(x - 100, y - 420) / 70 | 0) % 4];
  });
  const q = detectCard(px, W, H);
  assert.ok(q, 'the card was not found');
  for (const t of [[200, 150], [470, 150], [470, 320], [200, 320]]) assert.ok(Math.min(...q!.map((p) => Math.hypot(p[0] - t[0], p[1] - t[1]))) < 20);
  // the words of the card lie inside it; words spread over the whole picture do not
  assert.equal(quadHoldsWords(q!, [[250, 180], [300, 200], [350, 240], [420, 300], [230, 290]], W, H), true);
  assert.equal(quadHoldsWords(q!, [[50, 40], [600, 60], [350, 240], [80, 440], [590, 430]], W, H), false);
});
