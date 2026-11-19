// Browser test of the Florence-2 second reader. Model files are served from a
// local copy (data/models) through request interception, so the test does not
// depend on network access to huggingface.co.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const url = process.argv[2] ?? 'http://localhost:4173/';
const sample = process.argv[3] ?? 'Specimen passport';
const local = path.resolve(import.meta.dirname, '../../data/models');
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.setRequestInterception(true);
page.on('request', (r) => {
  const m = r.url().match(/huggingface\.co\/(onnx-community\/Florence-2-base-ft)\/resolve\/main\/(.+)$/);
  if (!m) return r.continue();
  const f = path.join(local, m[1], decodeURIComponent(m[2]));
  if (!fs.existsSync(f)) return r.respond({ status: 404, body: 'missing' });
  r.continue({ url: new URL('__models/' + m[1] + '/' + m[2], url).href });
});
await page.setViewport({ width: 1366, height: 900 });
await page.goto(url, { waitUntil: 'networkidle0' });
await page.evaluate(() => localStorage.setItem('ff.second', 'true'));
await page.reload({ waitUntil: 'networkidle0' });
await page.waitForFunction(() => /Ready on|Could not load/.test(document.body.textContent), { timeout: 300000 });

const t = Date.now();
await (await page.$$('input[type=file]'))[1].uploadFile(path.resolve(import.meta.dirname, '../public/samples/specimen-visa.jpg'));
await page.waitForSelector('.mini', { timeout: 120000 });
await page.click('.mini');
await page.waitForFunction(() => /Read twice|Second read failed/.test(document.body.innerText), { timeout: 600000 });
console.log('second read done in', Date.now() - t, 'ms');
console.log(await page.evaluate(() => [...document.querySelectorAll('.field')].map((f) => `${f.querySelector('.key').value} => ${f.querySelector('.val').value} [${f.querySelector('.conf')?.textContent}] ${[...f.querySelectorAll('.tag')].map((x) => x.textContent).join(', ')} ${f.querySelector('.alt q')?.textContent ? 'ALT: ' + f.querySelector('.alt q').textContent : ''}`).join('\n')));
console.log('ring:', await page.evaluate(() => document.querySelector('.trust .ring').getAttribute('aria-label')));
await page.screenshot({ path: '/tmp/shots/florence-visa.png', fullPage: true });
console.log(logs.filter((l) => /error|Error/.test(l) && !/Estimating|diacritics|CleanUnused/.test(l)).slice(0, 15).join('\n'));
await browser.close();
