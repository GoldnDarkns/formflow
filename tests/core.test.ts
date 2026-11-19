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
