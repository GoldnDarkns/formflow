"""Build the extra test set the brief asks for: 20 public forms filled in with invented data.

Blank forms: US federal forms (IRS, OPM), public domain, downloaded to ../data/forms20/blank.
Every value typed into a form is invented (seeded random); no real person appears.
Odd-numbered forms are also degraded to look like an office scan (tilt, blur, noise, JPEG).
About a third of the checkboxes are ticked and about half of the signature lines carry a
synthetic pen scribble (not a real signature).

Output (committed): testsets/forms20/<form>.jpg and truth.json with the value typed into
each box and where it is on the page image.

usage: python scripts/make-forms20.py      (needs PyMuPDF, Pillow, numpy)
"""
import io
import json
import os
import random
import re

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = os.path.join(HERE, '..', '..', 'data', 'forms20', 'blank')
OUT = os.path.join(HERE, '..', 'testsets', 'forms20')
BLANK_IMG = os.path.join(HERE, '..', '..', 'data', 'forms20', 'blank-img')
FORMS = ['f14039', 'f2553', 'f2848', 'f3911', 'f4506', 'f4506t', 'f4868', 'f56', 'f8332', 'f8821',
         'f8822', 'f8822b', 'f8888', 'f9465', 'fss4', 'fw4', 'fw7', 'fw8ben', 'fw9', 'sf1152']
SOURCE = {f: f'https://www.irs.gov/pub/irs-pdf/{f}.pdf' for f in FORMS}
SOURCE['sf1152'] = 'https://www.opm.gov/forms/pdf_fill/sf1152.pdf'
DPI = 150

FIRST = ['Mira', 'Tobias', 'Leona', 'Casper', 'Ingrid', 'Rafael', 'Nadia', 'Felix', 'Odette', 'Marcus', 'Selma', 'Jonas']
LAST = ['Halvorsen', 'Quintero', 'Ashdown', 'Belmore', 'Castellan', 'Draycott', 'Fenwick', 'Lindqvist', 'Marlowe', 'Okonjo', 'Pemberton', 'Vasquez']
STREET = ['Larkspur Lane', 'Quarry Road', 'Wexford Avenue', 'Juniper Court', 'Harbour Street', 'Maple Terrace', 'Orchard Way', 'Kestrel Drive']
CITY = [('Fairhaven', 'OH', '44012'), ('Brookfield', 'WI', '53005'), ('Easton', 'PA', '18042'), ('Glenwood', 'IA', '51534'), ('Ashland', 'OR', '97520'), ('Milford', 'CT', '06460')]
COMPANY = ['Northgate Supply Co', 'Bluewater Consulting LLC', 'Pinecrest Bakery Inc', 'Summit Ridge Partners', 'Lantern Press Ltd', 'Harrow and Vine LLC']
WORDS = ['Wholesale', 'Consulting', 'Retail', 'Sole proprietor', 'Manager', 'Treasurer', 'Owner', 'Spouse', 'Executor', 'Partner']


def invent(label, rng, max_len, width_pt):
    """An invented value that suits the box's label and size."""
    t = label.lower()
    digits = lambda n: ''.join(rng.choice('0123456789') for _ in range(n))
    if re.search(r'social security|ssn|taxpayer identification|\btin\b|itin', t):
        return f'{digits(3)}-{digits(2)}-{digits(4)}'
    if re.search(r'employer identification|\bein\b', t):
        return f'{digits(2)}-{digits(7)}'
    if max_len and max_len <= 11 and not re.search(r'name|city|state|title', t):
        return digits(max_len)
    if re.search(r'phone|telephone|fax', t):
        return f'({digits(3)}) {digits(3)}-{digits(4)}' if width_pt > 80 else f'{digits(3)}-{digits(4)}'
    if re.search(r'date|birth|\bdob\b|mm/dd', t):
        return f'{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1960, 2026)}'
    if re.search(r'zip|postal', t) and not re.search(r'city|town', t):
        return rng.choice(CITY)[2]
    if re.search(r'city|town', t):
        c = rng.choice(CITY)
        return f'{c[0]}, {c[1]} {c[2]}' if width_pt > 150 else c[0]
    if re.search(r'\bstate\b', t) and width_pt < 80:
        return rng.choice(CITY)[1]
    if re.search(r'address|street|apt|suite', t):
        return f'{rng.randint(12, 9800)} {rng.choice(STREET)}' if width_pt > 110 else str(rng.randint(1, 60))
    if re.search(r'business|company|corporation|organi[sz]ation|employer|entity|firm|trade name', t):
        return rng.choice(COMPANY)
    if re.search(r'first name', t):
        return rng.choice(FIRST)
    if re.search(r'last name|surname|family name', t):
        return rng.choice(LAST)
    if re.search(r'name|taxpayer|spouse|representative|appointee|fiduciary|person', t):
        return f'{rng.choice(FIRST)} {rng.choice(LAST)}' if width_pt > 90 else rng.choice(LAST)
    if re.search(r'amount|\$|payment|balance|total|wages|tax|income|percent|number of|year|period|line \d', t) or width_pt < 60:
        return str(rng.randint(1, 9)) if width_pt < 30 else str(rng.randint(120, 98000))
    if re.search(r'e-?mail', t):
        return f'{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@example.org'
    if re.search(r'title|relationship|capacity|type|occupation|reason', t):
        return rng.choice(WORDS)
    return rng.choice(WORDS) if width_pt < 140 else f'{rng.choice(FIRST)} {rng.choice(LAST)}'


def printed_label(words, r):
    """The text printed just above the box (same column), else just left of it on the same line."""
    above = [w for w in words if r.x0 - 3 <= (w[0] + w[2]) / 2 <= r.x1 and r.y0 - 13 <= (w[1] + w[3]) / 2 <= r.y0 + 1.5]
    if above:
        above.sort(key=lambda w: (round(w[1] / 4), w[0]))
        return ' '.join(w[4] for w in above)[:90]
    left = [w for w in words if r.y0 - 1 <= (w[1] + w[3]) / 2 <= r.y1 + 1 and w[2] <= r.x0 + 2 and r.x0 - w[2] < 260]
    left.sort(key=lambda w: w[0])
    return ' '.join(w[4] for w in left[-8:])[:90]


def signature_spots(words, page_w=612):
    """Where a person would sign: beside each printed 'Signature...' caption if the line is free, else just above it."""
    spots = []
    for w in words:
        if not w[4].lower().startswith('signature'):
            continue
        x0, y0, x1, y1 = w[:4]
        h = y1 - y0
        # the caption runs on while words follow closely on the same line
        line = sorted([v for v in words if abs((v[1] + v[3]) / 2 - (y0 + y1) / 2) < h * 0.6 and v[0] >= x1 - 1], key=lambda v: v[0])
        end = x1
        for v in line:
            if v[0] - end < 9:
                end = max(end, v[2])
        free = not any(end + 4 < v[0] < end + 135 for v in line)
        if any(v[4] in ('Part', 'Section') and abs((v[1] + v[3]) / 2 - (y0 + y1) / 2) < h and 0 < x0 - v[2] < 60 for v in words):
            continue  # a section heading, not a signing line
        # small captions sit under the signing line; larger ones stand left of it
        box = [end + 10, y0 - 12, end + 125, y1 + 3] if (free and h >= 8.5) else [x0 + 4, y0 - 25, x0 + 120, y0 - 3]
        shift = max(0, box[2] - (page_w - 30))
        box = [box[0] - shift, box[1], box[2] - shift, box[3]]
        if any(abs(b['caption'][1] - y0) < 30 and abs(b['caption'][0] - x0) < 200 for b in spots):
            continue  # one signing place per line
        spots.append({'caption': [x0, y0, x1, y1], 'box': box})
    return spots


def scribble(draw, box, rng):
    """A signature-like pen stroke: a wavy line with loops, drawn left to right inside the box."""
    x0, y0, x1, y1 = box
    mid, amp = (y0 + y1) / 2, (y1 - y0) / 2
    pts, x = [], x0
    phase, freq = rng.uniform(0, 6.28), rng.uniform(0.09, 0.16)
    import math
    i = 0
    while x < x1:
        a = amp * (0.35 + 0.65 * abs(math.sin(i * 0.11 + phase)))
        pts.append((x + 5 * math.sin(i * 0.5 + phase), mid + a * math.sin(i * freq * 6.28 / 2.2 + phase) * (1 if i % 23 < 17 else 0.3)))
        x += rng.uniform(1.2, 2.4)
        i += 1
    draw.line(pts, fill=rng.randint(15, 60), width=2, joint='curve')
    # a flourish under the name
    draw.line([(x0 + (x1 - x0) * 0.1, y1 - 2), (x0 + (x1 - x0) * rng.uniform(0.7, 0.95), y1 - rng.uniform(2, 7))], fill=40, width=2)


def printed_right(words, r):
    """The caption printed just right of a checkbox, on the same line."""
    right = sorted([w for w in words if r.y0 - 2 <= (w[1] + w[3]) / 2 <= r.y1 + 2 and w[0] >= r.x1 - 1 and w[0] - r.x1 < 200], key=lambda w: w[0])
    out, end = [], r.x1
    for w in right:
        if w[0] - end > 14:
            break
        out.append(w[4])
        end = w[2]
    return ' '.join(out)[:60]


def degrade(img, rng):
    """Make a clean render look like an office scan."""
    g = img.convert('L')
    g = g.rotate(rng.uniform(-1.4, 1.4), resample=Image.BICUBIC, expand=False, fillcolor=255)
    g = g.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 0.9)))
    a = np.asarray(g, dtype=np.float32)
    nrng = np.random.default_rng(rng.randint(0, 10**9))
    a = a * rng.uniform(0.9, 0.97) + nrng.normal(0, 7, a.shape)  # greyer paper + sensor noise
    a[nrng.random(a.shape) < 0.0006] = 0  # specks
    g = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    g.save(buf, 'JPEG', quality=45)
    return Image.open(io.BytesIO(buf.getvalue())).convert('L')


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = {'about': 'Invented data typed into public US federal forms. No real person appears.', 'dpi': DPI, 'forms': []}
    for n, form in enumerate(FORMS):
        rng = random.Random(1000 + n)
        doc = fitz.open(os.path.join(BLANK, form + '.pdf'))
        page = doc[0]
        k = DPI / 72
        fields = []
        words = page.get_text('words')
        # the blank page, for teaching a template (not committed)
        os.makedirs(BLANK_IMG, exist_ok=True)
        bp = page.get_pixmap(dpi=DPI, annots=True)
        Image.frombytes('RGB', (bp.width, bp.height), bp.samples).convert('L').save(os.path.join(BLANK_IMG, form + '.jpg'), 'JPEG', quality=85)
        for w in page.widgets():
            if w.field_type_string != 'Text' or (w.field_flags & 1):  # skip read-only
                continue
            r = w.rect
            if r.width < 14 or r.height < 7:
                continue
            tip = (w.field_label or '').strip()
            label = printed_label(words, r)
            max_len = w.text_maxlen or 0
            v = invent(label or tip, rng, max_len, r.width)
            # keep the text inside its box (about 5 pt per character at 9 pt Helvetica)
            fit = max(1, int(r.width / 5.0))
            if len(v) > fit:
                v = v[:fit].rstrip(' ,-(')
            if max_len:
                v = v[:max_len]
            if not v.strip():
                continue
            w.field_value = v
            w.text_fontsize = 9 if r.height >= 11 else 7
            w.update()
            fields.append({'label': label, 'tooltip': tip, 'value': v, 'box': [round(r.x0 * k), round(r.y0 * k), round(r.x1 * k), round(r.y1 * k)]})
        # tick about a third of the checkboxes (own random stream, so the typed values above never change)
        crng = random.Random(5000 + n)
        checks = []
        for w in page.widgets():
            if w.field_type_string != 'CheckBox' or (w.field_flags & 1):
                continue
            r = w.rect
            if r.width < 5 or r.height < 5 or r.width > 20 or abs(r.width - r.height) > 4:
                continue
            on = crng.random() < 0.35
            if on:
                w.field_value = w.on_state()
                w.update()
            checks.append({'label': printed_right(words, r), 'checked': on, 'box': [round(r.x0 * k), round(r.y0 * k), round(r.x1 * k), round(r.y1 * k)]})
        pix = page.get_pixmap(dpi=DPI, annots=True)
        img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples).convert('L')
        # sign about half of the signature lines with a synthetic scribble
        srng = random.Random(9000 + n)
        signatures = []
        draw = ImageDraw.Draw(img)
        for sp in signature_spots(words, page.rect.width):
            signed = srng.random() < 0.5
            b = [v * k for v in sp['box']]
            if signed:
                scribble(draw, b, srng)
            signatures.append({'signed': signed, 'caption': [round(v * k) for v in sp['caption']], 'box': [round(v) for v in b]})
        scan = n % 2 == 1
        if scan:
            img = degrade(img, rng)
        img.save(os.path.join(OUT, form + '.jpg'), 'JPEG', quality=55 if scan else 78)
        truth['forms'].append({'file': form + '.jpg', 'source': SOURCE[form], 'page': 1, 'quality': 'scan-like' if scan else 'clean', 'width': img.width, 'height': img.height, 'fields': fields, 'checkboxes': checks, 'signatures': signatures})
        print(form, 'scan' if scan else 'clean', len(fields), 'values', len(checks), 'checkboxes', sum(c['checked'] for c in checks), 'ticked', len(signatures), 'signature lines', sum(x['signed'] for x in signatures), 'signed')
    with open(os.path.join(OUT, 'truth.json'), 'w', encoding='utf8', newline='\n') as f:
        json.dump(truth, f, indent=1)
    print('total values', sum(len(x['fields']) for x in truth['forms']), 'checkboxes', sum(len(x['checkboxes']) for x in truth['forms']), 'signature lines', sum(len(x['signatures']) for x in truth['forms']))


if __name__ == '__main__':
    main()
