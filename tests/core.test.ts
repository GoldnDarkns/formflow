// Unit tests for the pure core (no browser needed). Run: npm test
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
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
