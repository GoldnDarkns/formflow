// Made-up Arabic test data, drawn by the browser (Edge/Chrome joins Arabic letters and lays them out
// right to left; the Python image library on this machine cannot). Everything is invented and every card is
// stamped SPECIMEN. No real card design is copied.
//   node scripts/make-arabic.mjs            -> ../data/arabic/{lines,cards}/*.png + truth.json
// lines: single text lines (Arabic, and English ones as negatives) for the reader's error rate and the
//        "is this Arabic?" decision. cards: bilingual mock ID cards for the whole Arabic path.
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const OUT = path.resolve('../data/arabic');
const BROWSER = process.env.BROWSER ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
let seed = +(process.env.SEED ?? 7);
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const pick = (a) => a[Math.floor(rnd() * a.length)];
const between = (a, b) => a + rnd() * (b - a);

const GIVEN = ['محمد', 'أحمد', 'علي', 'عمر', 'خالد', 'سارة', 'فاطمة', 'مريم', 'يوسف', 'حسن', 'نورة', 'ليلى', 'عبدالله', 'سلمان', 'هند', 'راشد', 'سعيد', 'منى', 'زينب', 'إبراهيم', 'طارق', 'هالة', 'جميلة', 'ماجد', 'نادية', 'كريم', 'رنا', 'سامي', 'وليد', 'عائشة'];
const FAMILY = ['الحسيني', 'العلوي', 'النجار', 'الخطيب', 'الصالح', 'البدري', 'القاسم', 'الراشد', 'المنصور', 'الحداد', 'العطار', 'السيد', 'الزين', 'النعيمي', 'الفارس', 'الهاشمي', 'الشريف', 'الطيب', 'الكبير', 'الأمين'];
const LATIN = { 'محمد': 'MOHAMMED', 'أحمد': 'AHMED', 'علي': 'ALI', 'عمر': 'OMAR', 'خالد': 'KHALID', 'سارة': 'SARA', 'فاطمة': 'FATIMA', 'مريم': 'MARYAM', 'يوسف': 'YOUSEF', 'حسن': 'HASSAN', 'نورة': 'NOURA', 'ليلى': 'LAILA', 'عبدالله': 'ABDULLA', 'سلمان': 'SALMAN', 'هند': 'HIND', 'راشد': 'RASHID', 'سعيد': 'SAEED', 'منى': 'MONA', 'زينب': 'ZAINAB', 'إبراهيم': 'IBRAHIM', 'طارق': 'TARIQ', 'هالة': 'HALA', 'جميلة': 'JAMILA', 'ماجد': 'MAJID', 'نادية': 'NADIA', 'كريم': 'KARIM', 'رنا': 'RANA', 'سامي': 'SAMI', 'وليد': 'WALEED', 'عائشة': 'AISHA', 'الحسيني': 'ALHUSSAINI', 'العلوي': 'ALALAWI', 'النجار': 'ALNAJJAR', 'الخطيب': 'ALKHATIB', 'الصالح': 'ALSALEH', 'البدري': 'ALBADRI', 'القاسم': 'ALQASSIM', 'الراشد': 'ALRASHED', 'المنصور': 'ALMANSOUR', 'الحداد': 'ALHADDAD', 'العطار': 'ALATTAR', 'السيد': 'ALSAYED', 'الزين': 'ALZAIN', 'النعيمي': 'ALNUAIMI', 'الفارس': 'ALFARES', 'الهاشمي': 'ALHASHMI', 'الشريف': 'ALSHARIF', 'الطيب': 'ALTAYEB', 'الكبير': 'ALKABIR', 'الأمين': 'ALAMEEN' };
const NATION = [['الإمارات العربية المتحدة', 'UNITED ARAB EMIRATES'], ['الهند', 'INDIA'], ['مصر', 'EGYPT'], ['الأردن', 'JORDAN'], ['باكستان', 'PAKISTAN'], ['الفلبين', 'PHILIPPINES'], ['المملكة المتحدة', 'UNITED KINGDOM'], ['سوريا', 'SYRIA'], ['لبنان', 'LEBANON'], ['المغرب', 'MOROCCO'], ['السودان', 'SUDAN'], ['عمان', 'OMAN']];
const LABELS_AR = ['الاسم', 'الجنسية', 'تاريخ الميلاد', 'تاريخ الانتهاء', 'رقم الهوية', 'الجنس', 'مكان الإصدار', 'تاريخ الإصدار'];
const LABELS_EN = ['Name', 'Nationality', 'Date of Birth', 'Expiry Date', 'ID Number', 'Sex', 'Place of Issue', 'Issuing Date'];
const FONTS = ['Arial', 'Tahoma', 'Segoe UI', 'Times New Roman'];

const name = () => {
  const n = 2 + Math.floor(rnd() * 3);
  const parts = [pick(GIVEN)];
  while (parts.length < n - 1) parts.push(pick(GIVEN));
  parts.push(pick(FAMILY));
  return parts;
};
const date = () => `${String(1 + Math.floor(rnd() * 28)).padStart(2, '0')}/${String(1 + Math.floor(rnd() * 12)).padStart(2, '0')}/${1960 + Math.floor(rnd() * 50)}`;
const idNo = () => `784-${1960 + Math.floor(rnd() * 50)}-${String(Math.floor(rnd() * 1e7)).padStart(7, '0')}-${Math.floor(rnd() * 10)}`;

function arabicLine() {
  const k = rnd();
  if (k < 0.45) return name().join(' ');
  if (k < 0.65) return pick(NATION)[0];
  if (k < 0.8) return pick(LABELS_AR);
  if (k < 0.92) return `${pick(['الاسم', 'الجنسية'])}: ${k < 0.86 ? name().join(' ') : pick(NATION)[0]}`;
  return pick(['دولة تجريبية', 'بطاقة هوية تجريبية', 'رخصة قيادة تجريبية', 'وزارة الداخلية التجريبية']);
}
function englishLine() {
  const k = rnd();
  if (k < 0.35) return name().map((p) => LATIN[p]).join(' ');
  if (k < 0.5) return date();
  if (k < 0.62) return idNo();
  if (k < 0.8) return pick(LABELS_EN);
  if (k < 0.9) return `${pick(['Name', 'Nationality'])}: ${pick(NATION)[1]}`;
  return pick(['SPECIMEN IDENTITY CARD', 'Resident Identity Card', 'Driving Licence']);
}

const style = (font, size, weight, blur, ink, bg) => `font-family:'${font}';font-size:${size}px;font-weight:${weight};filter:blur(${blur}px);color:${ink};background:${bg};`;

async function main() {
  const browser = await puppeteer.launch({ executablePath: BROWSER, headless: true, args: ['--no-sandbox', '--font-render-hinting=none'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 1 });

  // ---- lines
  const lines = [];
  const N = +(process.env.LINES ?? 240);
  for (let i = 0; i < N; i++) {
    const ar = i % 3 !== 2; // two Arabic lines for each English one
    lines.push({ file: `l${String(i).padStart(3, '0')}.png`, text: ar ? arabicLine() : englishLine(), script: ar ? 'ar' : 'en', font: pick(FONTS), size: Math.round(between(18, 40)), weight: rnd() < 0.35 ? 700 : 400, blur: rnd() < 0.5 ? 0 : between(0.2, 0.9), ink: pick(['#111', '#1c2333', '#333', '#0b3d2e']), bg: pick(['#fff', '#f4f1e8', '#e9eef2', 'repeating-linear-gradient(45deg,#f3efe6 0 6px,#e8e2d4 6px 12px)']) });
  }
  fs.mkdirSync(path.join(OUT, 'lines'), { recursive: true });
  for (const l of lines) {
    await page.setContent(`<html><body style="margin:0;padding:20px;background:#fff"><div id="l" dir="${l.script === 'ar' ? 'rtl' : 'ltr'}" style="display:inline-block;padding:${Math.round(l.size * 0.25)}px ${Math.round(l.size * 0.35)}px;white-space:nowrap;${style(l.font, l.size, l.weight, l.blur, l.ink, l.bg)}">${l.text}</div></body></html>`);
    const el = await page.$('#l');
    await el.screenshot({ path: path.join(OUT, 'lines', l.file) });
  }
  fs.writeFileSync(path.join(OUT, 'lines', 'truth.json'), JSON.stringify(lines.map(({ file, text, script, font, size, blur }) => ({ file, text, script, font, size, blur })), null, 1));

  // ---- cards (mock layout, two arrangements: stacked lines, or English left and Arabic right on one row)
  const cards = [];
  const C = +(process.env.CARDS ?? 40);
  fs.mkdirSync(path.join(OUT, 'cards'), { recursive: true });
  for (let i = 0; i < C; i++) {
    const layout = i % 2 ? 'side' : 'stacked', nat = pick(NATION), font = pick(FONTS);
    let n = name();
    while (layout === 'side' && n.length > 3) n = name();
    const c = { file: `c${String(i).padStart(3, '0')}.png`, nameAr: n.join(' '), nameEn: n.map((p) => LATIN[p]).join(' '), natAr: nat[0], natEn: nat[1], dob: date(), exp: date(), id: idNo(), font, layout, blur: rnd() < 0.5 ? 0 : between(0.2, 0.8) };
    const fs1 = Math.round(between(19, 26));
    const row = (en, ar) => layout === 'side'
      ? `<div style="display:flex;justify-content:space-between;gap:24px;margin:10px 0;white-space:nowrap;font-size:${fs1 - 4}px"><span>${en}</span><span dir="rtl">${ar}</span></div>`
      : `<div dir="rtl" style="margin:8px 0">${ar}</div><div style="margin:8px 0">${en}</div>`;
    const html = `<html><body style="margin:0;padding:30px;background:#777">
      <div id="c" style="width:1010px;height:640px;box-sizing:border-box;padding:28px 34px;border-radius:26px;position:relative;font-family:'${font}';font-size:${fs1}px;color:#1d2230;filter:blur(${c.blur}px);
        background:${pick(['#f2efe6', '#eaf0f2', 'repeating-linear-gradient(120deg,#f1ede2 0 9px,#e9e3d5 9px 18px)', '#f6f3ee'])}">
        <div style="display:flex;justify-content:space-between;font-weight:700;font-size:${fs1 + 4}px"><span>SPECIMEN IDENTITY CARD</span><span dir="rtl">بطاقة هوية تجريبية</span></div>
        <div style="display:flex;gap:30px;margin-top:26px">
          <div style="width:190px;height:240px;background:#c9c6bf;flex:none"></div>
          <div style="flex:1">
            ${row(`Name: ${c.nameEn}`, `الاسم: ${c.nameAr}`)}
            ${row(`Nationality: ${c.natEn}`, `الجنسية: ${c.natAr}`)}
            <div style="margin:10px 0">ID Number: ${c.id}</div>
            <div style="margin:10px 0">Date of Birth: ${c.dob}</div>
            <div style="margin:10px 0">Expiry Date: ${c.exp}</div>
          </div>
        </div>
        <div style="position:absolute;right:40px;bottom:30px;color:rgba(200,30,30,.55);font:700 48px Arial;transform:rotate(-8deg)">SPECIMEN</div>
      </div></body></html>`;
    await page.setContent(html);
    await (await page.$('#c')).screenshot({ path: path.join(OUT, 'cards', c.file) });
    cards.push(c);
  }
  fs.writeFileSync(path.join(OUT, 'cards', 'truth.json'), JSON.stringify(cards, null, 1));
  await browser.close();
  console.log('lines', lines.length, 'cards', cards.length, '->', OUT);
}
main();
