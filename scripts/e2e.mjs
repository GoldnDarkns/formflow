// Headless browser test of the built app: drop the sample scans, check the
// sorting, capture screenshots. usage: node scripts/e2e.mjs [url] [outDir]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const url = process.argv[2] ?? 'http://localhost:4173/';
const out = process.argv[3] ?? '/tmp/shots';
fs.mkdirSync(out, { recursive: true });
const EXPECT = { 'Scan 1': 'Visa or permit', 'Scan 2': 'Form', 'Scan 3': 'Emirates ID', 'Scan 4': 'Passport', 'Scan 5': 'Driving licence', 'Scan 6': 'Emirates ID', 'Scan 7': 'Form', 'Phone photo': 'Form' };
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.setRequestInterception(true);
page.on('request', (r) => {
  const m = r.url().match(/huggingface\.co\/(Xenova\/mobileclip_s0)\/resolve\/main\/(.+)$/);
  return m ? r.continue({ url: new URL('__models/' + m[1] + '/' + m[2], url).href }) : r.continue();
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clickText = (t) => page.evaluate((t) => [...document.querySelectorAll('button')].find((b) => b.textContent.includes(t)).click(), t);

await page.setViewport({ width: 1366, height: 900 });
await page.goto(url, { waitUntil: 'networkidle0' });
await sleep(900);
await page.screenshot({ path: `${out}/1-scan.png` });
const t0 = Date.now();
await clickText('sample scans');
await page.waitForSelector('.cutout', { timeout: 60000 });
await page.waitForFunction(() => document.querySelectorAll('.mini').length >= 2, { timeout: 120000 });
await sleep(700);
await page.screenshot({ path: `${out}/2-reading.png` });
await page.waitForFunction(() => document.querySelectorAll('.mini').length === 8, { timeout: 240000 });
await page.waitForSelector('.cutout svg', { timeout: 60000 });
await sleep(380);
await page.screenshot({ path: `${out}/2b-cutout.png` });
await page.waitForFunction(() => document.querySelectorAll('.mini').length === 9 && !document.querySelector('.cutout'), { timeout: 240000 });
await sleep(2500);
console.log('9 documents in', ((Date.now() - t0) / 1000).toFixed(1), 's');
await clickText('Open the trays');
await sleep(1600);
await page.screenshot({ path: `${out}/3-trays.png`, fullPage: true });
const cards = await page.evaluate(() => [...document.querySelectorAll('section')].flatMap((s) => [...s.querySelectorAll('.doc')].map((d) => ({ tray: s.querySelector('h2').textContent, file: d.querySelector('.tiny')?.textContent ?? d.querySelector('.who').textContent, who: d.querySelector('.who').textContent, stamp: d.querySelector('.stamp small').textContent, trust: d.querySelector('.ring').getAttribute('aria-label'), tags: [...d.querySelectorAll('.tag')].map((t) => t.textContent) }))));

let correct = 0;
for (const c of cards) { const ok = EXPECT[c.file] === c.tray; correct += ok; console.log(ok ? 'OK ' : 'BAD', c.file, '->', c.tray, c.stamp, '|', c.trust, '|', c.who, '|', c.tags.join('; ')); }
console.log('packets:', await page.evaluate(() => [...document.querySelectorAll('.packet')].map((p) => p.innerText.replace(/\n+/g, ' | ')).join('\n')));
fs.writeFileSync(path.resolve(import.meta.dirname, '../../data/specimens.json'), JSON.stringify({ correct, total: cards.length, cards }));

for (const [file, shot] of [['Scan 4', '4-passport'], ['Phone photo', '5-phone-photo'], ['Scan 5', '6-licence']]) {
  await page.evaluate((f) => [...document.querySelectorAll('.doc')].find((d) => d.innerText.includes(f)).click(), file);
  await page.waitForSelector('.results');
  await sleep(500);
  await page.evaluate(() => document.querySelector('.field')?.click());
  await sleep(1300);
  console.log(`\n== ${file} facts:`, await page.evaluate(() => [...document.querySelectorAll('.fact')].map((f) => f.innerText.replace(/\n/g, '=')).join('; ')));
  console.log('critic:', await page.evaluate(() => document.querySelector('.verdict')?.textContent + ' | ' + [...document.querySelectorAll('.duel > div')].map((d) => d.innerText.replace(/\n/g, ' ')).join(' | ')));
  console.log('alerts:', await page.evaluate(() => [...document.querySelectorAll('.alerts .tag')].map((f) => f.textContent).join('; ')));
  await page.screenshot({ path: `${out}/${shot}.png`, fullPage: true });
  await clickText('Back to trays');
  await sleep(400);
}
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
await sleep(500);
await page.screenshot({ path: `${out}/7-mobile-trays.png` });
await page.evaluate(() => [...document.querySelectorAll('.doc')].find((d) => d.innerText.includes('Scan 5')).click());
await sleep(1500);
await page.screenshot({ path: `${out}/8-mobile-detail.png` });
await page.evaluate(() => document.querySelectorAll('.nav button')[0].click());
await sleep(900);
await page.screenshot({ path: `${out}/9-mobile-scan.png` });
console.log('\n--- console ---\n' + logs.filter((l) => !/Estimating|diacritics|debug|Invalid resolution/.test(l)).slice(0, 20).join('\n'));
await browser.close();
