"""Generate made-up specimen documents for training and testing FormFlow.

Every document is fictional: invented names, random numbers, a grey silhouette
instead of a face, and the word SPECIMEN printed on it. The layouts are generic
(they follow the public ICAO 9303 structure for machine-readable lines), not
copies of any country's real design, and carry no emblems or security features,
so nothing here could pass as a real document. What they do share with real
documents is what a reader has to cope with: many countries and languages,
label/value layouts, code lines with correct check digits, and photos taken
sideways, at an angle, blurred or with glare.

Types: passport, id_card, emirates_id (front/back), driving_licence (card and
international permit), vehicle_licence, visa.

Output (outside the repository): ../data/synth/<split>/<type>/<name>.jpg and
truth.json per split with the true type and field values.

usage: python scripts/make-specimens.py [per_type_train=36] [per_type_test=18]
"""
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'data', 'synth')
FONTS = 'C:/Windows/Fonts/'
SANS = ['arial.ttf', 'calibri.ttf', 'verdana.ttf', 'tahoma.ttf', 'ARIALN.TTF']
SANS_B = ['arialbd.ttf', 'calibrib.ttf', 'verdanab.ttf', 'tahomabd.ttf', 'ARIALNB.TTF']
SERIF = ['times.ttf', 'georgia.ttf']
MONO = ['consola.ttf', 'cour.ttf', 'courbd.ttf', 'consolab.ttf']


def font(name, size):
    return ImageFont.truetype(FONTS + name, int(size))


# country: ISO code, name, "passport" word, "driving licence" words, nationality. Split so some are only ever in the test set.
COUNTRIES = [
    ('BRA', 'FEDERATIVE REPUBLIC OF BRAZIL', 'PASSAPORTE / PASSPORT', 'CARTEIRA NACIONAL DE HABILITAÇÃO / DRIVER LICENSE', 'BRAZILIAN'),
    ('SRB', 'REPUBLIC OF SERBIA', 'PASOŠ / PASSPORT', 'VOZAČKA DOZVOLA / DRIVING LICENCE', 'SERBIAN'),
    ('ROU', 'ROMANIA', 'PAȘAPORT / PASSPORT', 'PERMIS DE CONDUCERE / DRIVING LICENCE', 'ROMANIAN'),
    ('RUS', 'RUSSIAN FEDERATION', 'ПАСПОРТ / PASSPORT', 'DRIVING LICENCE', 'RUSSIAN'),
    ('PAK', 'ISLAMIC REPUBLIC OF PAKISTAN', 'PASSPORT', 'DRIVING LICENCE', 'PAKISTANI'),
    ('AUT', 'REPUBLIC OF AUSTRIA', 'REISEPASS / PASSPORT', 'FÜHRERSCHEIN / DRIVING LICENCE', 'AUSTRIAN'),
    ('GBR', 'UNITED KINGDOM', 'PASSPORT', 'DRIVING LICENCE', 'BRITISH'),
    ('IND', 'REPUBLIC OF INDIA', 'PASSPORT', 'DRIVING LICENCE', 'INDIAN'),
    ('FRA', 'RÉPUBLIQUE FRANÇAISE', 'PASSEPORT / PASSPORT', 'PERMIS DE CONDUIRE / DRIVING LICENCE', 'FRENCH'),
    ('ESP', 'REINO DE ESPAÑA', 'PASAPORTE / PASSPORT', 'PERMISO DE CONDUCCIÓN / DRIVING LICENCE', 'SPANISH'),
    ('DEU', 'BUNDESREPUBLIK DEUTSCHLAND', 'REISEPASS / PASSPORT', 'FÜHRERSCHEIN / DRIVING LICENCE', 'GERMAN'),
    ('PHL', 'REPUBLIC OF THE PHILIPPINES', 'PASAPORTE / PASSPORT', "DRIVER'S LICENSE", 'FILIPINO'),
    ('EGY', 'ARAB REPUBLIC OF EGYPT', 'PASSPORT', 'DRIVING LICENCE', 'EGYPTIAN'),
    ('NGA', 'FEDERAL REPUBLIC OF NIGERIA', 'PASSPORT', "DRIVER'S LICENCE", 'NIGERIAN'),
    ('USA', 'UNITED STATES OF AMERICA', 'PASSPORT', 'DRIVER LICENSE', 'AMERICAN'),
    ('CAN', 'CANADA', 'PASSPORT / PASSEPORT', "DRIVER'S LICENCE / PERMIS DE CONDUIRE", 'CANADIAN'),
    # the next six appear only in the test set
    ('ITA', 'REPUBBLICA ITALIANA', 'PASSAPORTO / PASSPORT', 'PATENTE DI GUIDA / DRIVING LICENCE', 'ITALIAN'),
    ('TUR', 'REPUBLIC OF TÜRKIYE', 'PASAPORT / PASSPORT', 'SÜRÜCÜ BELGESİ / DRIVING LICENCE', 'TURKISH'),
    ('ZAF', 'REPUBLIC OF SOUTH AFRICA', 'PASSPORT', 'DRIVING LICENCE', 'SOUTH AFRICAN'),
    ('AUS', 'AUSTRALIA', 'PASSPORT', 'DRIVER LICENCE', 'AUSTRALIAN'),
    ('JPN', 'JAPAN', 'PASSPORT', 'DRIVING LICENCE', 'JAPANESE'),
    ('KEN', 'REPUBLIC OF KENYA', 'PASSPORT', 'DRIVING LICENCE', 'KENYAN'),
]
TEST_ONLY = 6
GIVEN = ['Tomas', 'Elena', 'Mirko', 'Aisha', 'Viktor', 'Leila', 'Rafael', 'Ingrid', 'Samir', 'Noemi', 'Kofi', 'Yuki', 'Dario', 'Priya', 'Anders', 'Camila', 'Oleg', 'Farah', 'Mateus', 'Hana',
         'Luca Marco', 'Ana Beatriz', 'Jean Paul', 'Maria Jose', 'Abdul Karim', 'Sofia Elena', 'Ravi Kumar', 'Emma Louise']
SURNAME = ['Novak', 'Almeida', 'Petrovic', 'Lindgren', 'Haddad', 'Okafor', 'Tanaka', 'Moreau', 'Castillo', 'Weber', 'Ionescu', 'Sokolov', 'Rahman', 'Mensah', 'Oliveira', 'Kowalski', 'Demir', 'Bianchi',
           'Fernandez', 'Nakamura', 'De Souza Lima', 'Van Der Berg', 'Al Mansoori', 'Garcia Lopez', 'Singh Chauhan']
CITIES = ['Novi Grad', 'Santa Clara', 'Port Elias', 'Kiruna Bay', 'Al Noor', 'Riverton', 'Monte Verde', 'Stonehaven', 'Lakeside', 'San Miro', 'Eastfield', 'Dunmore']
MAKES = ['NISSAN SUNNY', 'TOYOTA YARIS', 'MG ZS', 'KIA PICANTO', 'HYUNDAI ACCENT', 'MITSUBISHI ATTRAGE', 'HONDA CITY', 'SUZUKI DZIRE', 'RENAULT SYMBOL', 'MAZDA 2']
COLOURS = ['WHITE', 'SILVER', 'BLUE', 'BLACK', 'RED', 'GREY']
INSURERS = ['NORTHSTAR INSURANCE', 'GULF SHIELD ASSURANCE', 'HARBOUR MUTUAL', 'ORIENT COVER CO', 'CRESCENT TAKAFUL']
OWNERS = ['SUMMIT RENT A CAR LLC', 'BLUE DUNE MOTORS LLC', 'CEDAR LEASING CO', 'FALCON AUTO RENTALS', 'PALM CITY CARS LLC']
JOBS = ['Accountant', 'Sales Executive', 'Engineer', 'Driver', 'Designer', 'Teacher', 'Technician', 'Manager', 'Chef', 'Nurse']


# ---------- ICAO 9303 machine-readable lines ----------
def cd(s):
    val = lambda c: 0 if c == '<' else int(c) if c.isdigit() else ord(c) - 55
    return str(sum(val(c) * (7, 3, 1)[i % 3] for i, c in enumerate(s)) % 10)


def mrz_name(sur, giv, n):
    f = lambda x: ''.join(c if c.isalpha() else '<' for c in x.upper().replace(' ', '<'))
    return (f(sur) + '<<' + f(giv)).ljust(n, '<')[:n]


def td3(code, num, nat, dob, sex, exp, sur, giv, personal=''):
    l1 = 'P<' + code + mrz_name(sur, giv, 39)
    num = num.ljust(9, '<')
    pers = personal.ljust(14, '<')
    body = num + cd(num) + nat + dob + cd(dob) + sex + exp + cd(exp) + pers + (cd(pers) if personal else '<')
    comp = body[0:10] + body[13:20] + body[21:43]
    return [l1, body + cd(comp)]


def td1(code, num, nat, dob, sex, exp, sur, giv, opt1='', opt2='', doc='I<'):
    num = num.ljust(9, '<')
    l1 = (doc + code + num + cd(num) + opt1).ljust(30, '<')[:30]
    l2 = dob + cd(dob) + sex + exp + cd(exp) + nat + opt2.ljust(11, '<')[:11]
    comp = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    return [l1, l2 + cd(comp), mrz_name(sur, giv, 30)]


def mrva(code, num, nat, dob, sex, exp, sur, giv):
    l1 = 'V<' + code + mrz_name(sur, giv, 39)
    num = num.ljust(9, '<')
    return [l1, (num + cd(num) + nat + dob + cd(dob) + sex + exp + cd(exp)).ljust(44, '<')]


def luhn_digit(digits):
    s = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        s += d
    return str((10 - s % 10) % 10)


# ---------- a made-up person ----------
def person(rng, countries):
    c = rng.choice(countries)
    y, m, d = rng.randint(1958, 2005), rng.randint(1, 12), rng.randint(1, 28)
    iy, im, idd = rng.randint(2019, 2026), rng.randint(1, 12), rng.randint(1, 28)
    ey = iy + rng.choice([2, 3, 5, 10])
    return {'code': c[0], 'country': c[1], 'passport_word': c[2], 'dl_word': c[3], 'nat': c[4], 'given': rng.choice(GIVEN), 'surname': rng.choice(SURNAME), 'sex': rng.choice('MF'),
            'dob': (y, m, d), 'issue': (iy, im, idd), 'expiry': (ey, im, max(1, idd - 1)), 'city': rng.choice(CITIES)}


def fmt(date, style):
    y, m, d = date
    mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][m - 1]
    return {'dmy/': f'{d:02d}/{m:02d}/{y}', 'dmy.': f'{d:02d}.{m:02d}.{y}', 'ymd-': f'{y}-{m:02d}-{d:02d}', 'dMy': f'{d:02d} {mon} {y}', 'dmy-': f'{d:02d}-{m:02d}-{y}'}[style]


iso = lambda date: f'{date[0]}-{date[1]:02d}-{date[2]:02d}'
yymmdd = lambda date: f'{date[0] % 100:02d}{date[1]:02d}{date[2]:02d}'
alnum = lambda rng, n, letters=2: ''.join(rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ') for _ in range(letters)) + ''.join(rng.choice('0123456789') for _ in range(n - letters))
digits = lambda rng, n: ''.join(rng.choice('0123456789') for _ in range(n))


# ---------- drawing ----------
class Card:
    def __init__(self, rng, w, h, tint=None):
        self.rng = rng
        base = tint or rng.choice([(236, 240, 246), (244, 240, 230), (232, 242, 236), (246, 236, 240), (238, 238, 232), (228, 236, 244), (250, 244, 214), (226, 230, 226)])
        self.ink = rng.choice([(22, 24, 30), (30, 30, 60), (40, 30, 30), (16, 40, 40)])
        self.accent = rng.choice([(150, 30, 40), (20, 60, 140), (10, 100, 80), (110, 40, 120), (60, 60, 60)])
        self.img = Image.new('RGB', (w, h), base)
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self.sans, self.bold = rng.choice(list(zip(SANS, SANS_B)))
        # a faint background pattern, like the fine print real documents have
        col = tuple(max(0, c - rng.randint(8, 16)) for c in base)
        step = rng.randint(10, 22)
        style = rng.choice(['lines', 'waves', 'grid', 'none'])
        if style == 'lines':
            for x in range(-h, w, step):
                self.d.line([(x, 0), (x + h, h)], fill=col, width=1)
        elif style == 'grid':
            for x in range(0, w, step):
                self.d.line([(x, 0), (x, h)], fill=col, width=1)
            for y in range(0, h, step):
                self.d.line([(0, y), (w, y)], fill=col, width=1)
        elif style == 'waves':
            for y0 in range(0, h, step):
                self.d.line([(x, y0 + 5 * math.sin(x / 23)) for x in range(0, w, 6)], fill=col, width=1)

    def text(self, xy, s, size, bold=False, fill=None, mono=False, serif=False, anchor='la'):
        f = font(self.rng.choice(MONO) if mono is True else mono if mono else (self.rng.choice(SERIF) if serif else (self.bold if bold else self.sans)), size)
        self.d.text(xy, s, font=f, fill=fill or self.ink, anchor=anchor)
        return self.d.textbbox(xy, s, font=f, anchor=anchor)

    def photo(self, box):
        x0, y0, x1, y1 = box
        self.d.rectangle(box, fill=(205, 210, 216), outline=(150, 156, 164))
        cx, w, h = (x0 + x1) / 2, x1 - x0, y1 - y0
        self.d.ellipse([cx - w * 0.2, y0 + h * 0.16, cx + w * 0.2, y0 + h * 0.5], fill=(168, 174, 182))
        self.d.pieslice([cx - w * 0.42, y0 + h * 0.56, cx + w * 0.42, y0 + h * 1.5], 180, 360, fill=(168, 174, 182))

    def specimen(self):
        s = self.rng.choice(['SPECIMEN', 'SPECIMEN · NOT A REAL DOCUMENT', 'SAMPLE · FICTIONAL DATA'])
        layer = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f = font('arialbd.ttf', self.h * (0.11 if len(s) < 10 else 0.055))
        d.text((self.w * 0.5, self.h * self.rng.uniform(0.42, 0.6)), s, font=f, fill=(200, 30, 30, 70), anchor='mm')
        layer = layer.rotate(self.rng.uniform(-22, -8), resample=Image.BICUBIC)
        self.img = Image.alpha_composite(self.img.convert('RGBA'), layer).convert('RGB')
        self.d = ImageDraw.Draw(self.img)
        self.text((self.w - 12, self.h - 8), 'SPECIMEN - fictional data', 11, fill=(170, 40, 40), anchor='rs')

    def fields(self, items, x, y, col_w, cols, label_size, value_size, gap, side=False, mono_values=False):
        """Label above value (or label left of value) in a grid."""
        vfont = self.rng.choice(MONO) if mono_values else None
        for i, (lab, val) in enumerate(items):
            cx, cy = x + (i % cols) * col_w, y + (i // cols) * gap
            if side:
                b = self.text((cx, cy), lab, label_size, fill=(90, 96, 104))
                self.text((cx + col_w * 0.42, cy - (value_size - label_size) * 0.6), val, value_size, bold=not vfont, mono=vfont)
            else:
                self.text((cx, cy), lab, label_size, fill=(90, 96, 104))
                self.text((cx, cy + label_size * 1.25), val, value_size, bold=not vfont, mono=vfont)

    def mrz(self, lines, y, size=None):
        f = self.rng.choice(MONO)
        size = size or int(self.w / (len(lines[0]) * 0.62))
        self.d.rectangle([0, y - size * 0.6, self.w, self.h], fill=tuple(min(255, c + 8) for c in self.img.getpixel((5, 5))))
        fo = font(f, size)
        tw = self.d.textlength(lines[0], font=fo)
        for i, line in enumerate(lines):
            self.d.text(((self.w - tw) / 2, y + i * size * 1.45), line, font=fo, fill=(15, 15, 18))


def lab(rng, en, alt):
    """Bilingual or English-only label, in the varied styles real documents use."""
    return rng.choice([en, f'{en} / {alt}', f'{alt} / {en}', en.upper(), f'{rng.randint(1, 9)}. {en}'])


def make_passport(rng, P):
    c = Card(rng, 1250, 880)
    num = alnum(rng, rng.choice([8, 9]), rng.choice([0, 1, 2]))
    ds = rng.choice(['dMy', 'dmy.', 'dmy/'])
    c.text((40, 26), P['country'], 30, bold=True, fill=c.accent)
    c.text((40, 66), P['passport_word'], 20)
    c.photo([44, 150, 330, 520])
    x = 370
    c.fields([(lab(rng, 'Type', 'Tipo'), 'P'), (lab(rng, 'Country code', 'Code du pays'), P['code']), (lab(rng, 'Passport No.', 'No. du passeport'), num)], x, 120, 280, 3, 14, 24, 70)
    c.fields([(lab(rng, 'Surname', 'Nom'), P['surname'].upper()), (lab(rng, 'Given names', 'Prénoms'), P['given'].upper()), (lab(rng, 'Nationality', 'Nationalité'), P['nat']),
              (lab(rng, 'Date of birth', 'Date de naissance'), fmt(P['dob'], ds)), (lab(rng, 'Sex', 'Sexe'), P['sex']), (lab(rng, 'Place of birth', 'Lieu de naissance'), P['city'].upper()),
              (lab(rng, 'Date of issue', 'Date de délivrance'), fmt(P['issue'], ds)), (lab(rng, 'Date of expiry', "Date d'expiration"), fmt(P['expiry'], ds)), (lab(rng, 'Authority', 'Autorité'), 'PASSPORT OFFICE')],
             x, 200, 420, 2, 14, 24, 72)
    lines = td3(P['code'], num, P['code'], yymmdd(P['dob']), P['sex'], yymmdd(P['expiry']), P['surname'], P['given'])
    c.mrz(lines, 730, 37)
    c.specimen()
    return c.img, {'Passport number': num, 'Date of birth': iso(P['dob']), 'Expiry date': iso(P['expiry']), 'Full name': f"{P['given']} {P['surname']}".upper(), 'mrz': lines}


def make_id_card(rng, P, back=None):
    back = rng.random() < 0.4 if back is None else back
    c = Card(rng, 1250, 790)
    num = alnum(rng, 9, rng.choice([0, 2]))
    lines = td1(P['code'], num, P['code'], yymmdd(P['dob']), P['sex'], yymmdd(P['expiry']), P['surname'], P['given'])
    ds = rng.choice(['dmy/', 'dmy.', 'dMy'])
    truth = {'Document number': num, 'Date of birth': iso(P['dob']), 'Expiry date': iso(P['expiry']), 'Full name': f"{P['given']} {P['surname']}".upper()}
    if back:
        c.fields([(lab(rng, 'Address', 'Adresse'), f"{rng.randint(1, 240)} {P['city']} Road"), (lab(rng, 'Issuing authority', 'Autorité'), 'CIVIL REGISTRY'), (lab(rng, 'Date of issue', 'Date de délivrance'), fmt(P['issue'], ds))], 60, 70, 560, 2, 16, 26, 90)
        c.mrz(lines, 470, 52)
        truth['mrz'] = lines
    else:
        c.text((44, 28), P['country'], 32, bold=True, fill=c.accent)
        c.text((44, 74), rng.choice(['NATIONAL IDENTITY CARD', 'IDENTITY CARD / CARTE D\'IDENTITÉ', 'NATIONAL ID', 'IDENTITY CARD']), 22)
        left = rng.random() < 0.7
        c.photo([44, 150, 310, 500] if left else [930, 150, 1196, 500])
        c.fields([(lab(rng, 'Surname', 'Nom'), P['surname'].upper()), (lab(rng, 'Given names', 'Prénoms'), P['given'].upper()), (lab(rng, 'Nationality', 'Nationalité'), P['nat']), (lab(rng, 'Date of birth', 'Date de naissance'), fmt(P['dob'], ds)),
                  (lab(rng, 'Sex', 'Sexe'), P['sex']), (lab(rng, 'ID Number', 'No.'), num), (lab(rng, 'Date of expiry', "Date d'expiration"), fmt(P['expiry'], ds))], 360 if left else 60, 150, 420, 2, 16, 27, 96)
    c.specimen()
    return c.img, truth


def make_emirates_id(rng, P, back=None):
    back = rng.random() < 0.45 if back is None else back
    c = Card(rng, 1250, 790, tint=rng.choice([(232, 238, 244), (238, 234, 244), (236, 242, 240)]))
    body = '784' + str(P['dob'][0]) + digits(rng, 7)
    eid = body + luhn_digit(body)
    eid_fmt = f'{eid[:3]}-{eid[3:7]}-{eid[7:14]}-{eid[14]}'
    card_no = digits(rng, 9)
    lines = td1('ARE', card_no, P['code'], yymmdd(P['dob']), P['sex'], yymmdd(P['expiry']), P['surname'], P['given'], opt1=eid, doc='IL')
    truth = {'Emirates ID number': eid_fmt, 'Date of birth': iso(P['dob']), 'Expiry date': iso(P['expiry']), 'Full name': f"{P['given']} {P['surname']}".upper()}
    if back:
        c.text((60, 40), 'Card Number', 22)
        c.text((60, 72), card_no, 28, bold=True)
        c.fields([('Occupation:', rng.choice(JOBS)), ('Employer:', rng.choice(OWNERS).title()), ('Issuing Place:', rng.choice(['Dubai', 'Sharjah', 'Abu Dhabi', 'Ajman']))], 380, 150, 800, 1, 22, 28, 70, side=True)
        c.mrz(lines, 520, 50)
        truth['mrz'] = lines
    else:
        c.text((44, 26), 'UNITED ARAB EMIRATES', 34, bold=True)
        c.text((44, 72), rng.choice(['Resident Identity Card', 'Identity Card', 'Resident Identity Card']), 22, bold=True)
        c.photo([60, 150, 320, 500])
        c.text((380, 130), 'ID Number', 22)
        c.text((380, 160), eid_fmt, 34, bold=True)
        ds = 'dmy/'
        c.fields([('Name:', f"{P['given']} {P['surname']}"), ('Date of Birth:', fmt(P['dob'], ds)), ('Nationality:', P['nat'].title()), ('Issuing Date', fmt(P['issue'], ds)), ('Expiry Date', fmt(P['expiry'], ds)), ('Sex:', P['sex'])], 380, 240, 760, 1, 22, 28, 66, side=True)
    c.specimen()
    return c.img, truth


def make_driving_licence(rng, P):
    kind = rng.choice(['table', 'numbered', 'permit', 'back'])
    num = digits(rng, rng.choice([6, 7, 9])) if rng.random() < 0.7 else alnum(rng, 10, 3)
    ds = rng.choice(['dmy/', 'dmy.', 'ymd-', 'dmy-'])
    truth = {'Licence number': num, 'Date of birth': iso(P['dob']), 'Expiry date': iso(P['expiry']), 'Full name': f"{P['given']} {P['surname']}".upper()}
    if kind == 'permit':  # international driving permit, booklet page
        c = Card(rng, 900, 1250, tint=(226, 230, 226))
        c.text((450, 60), P['country'], 30, bold=True, anchor='ma')
        c.text((450, 130), 'INTERNATIONAL MOTOR TRAFFIC', 24, bold=True, anchor='ma')
        c.text((450, 230), 'INTERNATIONAL DRIVING PERMIT', 36, serif=True, anchor='ma')
        c.text((450, 290), 'No ' + alnum(rng, 8, 2), 34, bold=True, fill=(190, 40, 40), anchor='ma')
        c.text((450, 350), 'PERMIS INTERNATIONAL DE CONDUIRE', 26, serif=True, anchor='ma')
        c.text((450, 400), 'Convention on Road Traffic of 8 November 1968', 20, anchor='ma')
        c.fields([('Valid until', fmt(P['expiry'], ds)), ('Issued by', rng.choice(['Automobile Association', 'National Motor Club', 'Drivers Association'])), ('Date of issue', fmt(P['issue'], ds)), ('Number of national licence', num),
                  ('1. Surname', P['surname'].upper()), ('2. Given names', P['given'].upper()), ('3. Place of birth', P['city']), ('4. Date of birth', fmt(P['dob'], ds))], 90, 480, 720, 1, 18, 26, 84, side=True)
    elif kind == 'back':  # back of a card: categories and a traffic code
        c = Card(rng, 1250, 790, tint=(226, 228, 232))
        c.d.rectangle([60, 50, 1190, 110], outline=c.ink, width=2)
        c.text((80, 66), 'Traffic Code No.', 22, bold=True)
        c.text((620, 60), digits(rng, 8), 34, bold=True, anchor='ma')
        c.d.rectangle([60, 130, 1190, 180], outline=c.ink, width=2)
        c.text((80, 142), 'Permitted Vehicles', 22, bold=True)
        cells = [(240, 240, 244), (170, 180, 220), (240, 220, 160), (200, 200, 200), (245, 240, 140), (240, 240, 244), (170, 210, 190), (250, 210, 225), (180, 220, 245), (225, 140, 130)]
        for i, col in enumerate(cells):
            x0, y0 = 60 + (i % 5) * 226, 200 + (i // 5) * 100
            c.d.rectangle([x0, y0, x0 + 226, y0 + 100], fill=col, outline=c.ink, width=2)
        c.text((60 + 3 * 226 + 113, 232), rng.choice(['Light Vehicle', 'Light Vehicle Automatic']), 22, bold=True, anchor='ma')
        if rng.random() < 0.5:
            c.text((60 + 4 * 226 + 113, 232), 'Motor Cycle', 22, bold=True, anchor='ma')
        c.text((70, 430), '• This license should be with you when you are driving and', 20, bold=True)
        c.text((70, 458), '  should be presented on demand to an authorized person.', 20, bold=True)
        c.text((70, 510), '• Give way to ambulance, civil defence and police vehicles.', 20, bold=True)
        truth = {}
    elif kind == 'table':  # label left, value right, in ruled rows
        c = Card(rng, 1250, 790, tint=(228, 230, 234))
        c.text((300, 50), rng.choice(['United Arab Emirates', P['country'].title()]), 30, serif=True)
        c.text((300, 110), rng.choice(['Driving License', 'Driving Licence', 'DRIVING LICENCE']), 30, bold=True, fill=(190, 40, 60))
        c.photo([60, 250, 300, 560])
        rows = [('License No.', num), ('Name', f"{P['given']} {P['surname']}".upper()), ('Nationality', P['nat']), ('Date of Birth', fmt(P['dob'], ds)), ('Issue Date', fmt(P['issue'], ds)), ('Expiry Date', fmt(P['expiry'], ds)), ('Place of Issue', rng.choice(['Dubai', 'Sharjah', P['city']]))]
        for i, (l, v) in enumerate(rows):
            y = 200 + i * 78
            c.d.line([(340, y), (1190, y)], fill=c.ink, width=1)
            c.text((356, y + 24), l, 22, bold=True)
            c.text((660, y + 18), v, 28, bold=rng.random() < 0.5)
        c.d.rectangle([340, 200, 1190, 200 + len(rows) * 78], outline=c.ink, width=2)
        c.text((70, 600), 'Licensing Authority', 16, bold=True)
    else:  # EU-style numbered fields
        c = Card(rng, 1250, 790, tint=(244, 230, 236))
        c.d.rectangle([40, 40, 200, 150], fill=(30, 60, 150))
        c.text((120, 95), P['code'], 44, bold=True, fill=(255, 255, 255), anchor='mm')
        c.text((230, 44), P['dl_word'], 26, bold=True, fill=(30, 60, 150))
        c.text((230, 86), P['country'], 22)
        c.photo([60, 200, 330, 560])
        rows = [('1.', P['surname'].upper()), ('2.', P['given'].upper()), ('3.', f"{fmt(P['dob'], ds)}  {P['city'].upper()}"), ('4a.', fmt(P['issue'], ds)), ('4b.', fmt(P['expiry'], ds)), ('4c.', 'TRANSPORT AUTHORITY'), ('5.', num), ('9.', rng.choice(['B', 'AM B', 'B BE', 'A B']))]
        for i, (l, v) in enumerate(rows):
            c.text((380, 190 + i * 62), l, 20)
            c.text((440, 184 + i * 62), v, 30, bold=True)
    c.specimen()
    return c.img, truth


def make_vehicle_licence(rng, P):
    back = rng.random() < 0.5
    c = Card(rng, 1250, 790, tint=(250, 244, 214))
    plate = f"{rng.choice('ABCDEFGHKLMN')}{rng.choice(['', rng.choice('ABCDE')])} / {digits(rng, 5)}"
    if back:
        c.text((50, 150), 'Vehicle Information', 36, serif=True)
        rows = [('Model', str(rng.randint(2016, 2026)), 'Num. of Pass.', str(rng.choice([2, 5, 7]))), ('Origin', rng.choice(['India', 'Japan', 'China', 'Korea', 'Thailand']), '', ''), ('Veh. Type', rng.choice(MAKES), 'Colour', rng.choice(COLOURS)),
                ('G. V. W.', str(rng.randint(1500, 2200)), 'Empty Weight', str(rng.randint(950, 1450))), ('Eng. No.', alnum(rng, 10, 4), '', ''), ('Chassis No.', alnum(rng, 17, 9), '', '')]
        truth = {'Chassis number': rows[5][1]}
    else:
        c.text((60, 80), 'UNITED ARAB EMIRATES', 34, serif=True)
        c.text((270, 140), 'Vehicle License', 32, bold=True, fill=(190, 40, 60))
        rows = [('Traffic Plate No.', plate, '', ''), ('Place of Issue', rng.choice(['Dubai', 'Sharjah', 'Ajman']), '', ''), ('T. C. No.', digits(rng, 8), '', ''), ('Owner', rng.choice(OWNERS), '', ''), ('Nationality', '', '', ''),
                ('Exp. Date', fmt(P['expiry'], 'ymd-'), 'Reg. Date', fmt(P['issue'], 'ymd-')), ('Ins. Exp.', fmt(P['expiry'], 'ymd-'), 'Insurer', rng.choice(INSURERS)), ('Policy No.', digits(rng, 10), '', ''), ('Mortgage By', '', '', '')]
        truth = {'Plate number': plate}
    y0 = 220
    rh = (790 - y0 - 60) / len(rows)
    for i, (l1, v1, l2, v2) in enumerate(rows):
        y = y0 + i * rh
        c.d.line([(40, y), (1210, y)], fill=c.ink, width=1)
        c.text((56, y + rh * 0.22), l1, 26, serif=True)
        c.text((270, y + rh * 0.18), v1, 28, bold=True)
        if l2:
            c.text((700, y + rh * 0.22), l2, 26, serif=True)
            c.text((930, y + rh * 0.18), v2, 28, bold=True)
    c.d.rectangle([40, y0, 1210, y0 + rh * len(rows)], outline=c.ink, width=2)
    c.d.line([(250, y0), (250, y0 + rh * len(rows))], fill=c.ink, width=1)
    c.specimen()
    return c.img, truth


def make_visa(rng, P):
    sticker = rng.random() < 0.45
    num = alnum(rng, 9, 1)
    ds = rng.choice(['dmy/', 'dMy', 'dmy.'])
    truth = {'Date of birth': iso(P['dob']), 'Full name': f"{P['given']} {P['surname']}".upper()}
    if sticker:
        c = Card(rng, 1250, 860)
        c.text((40, 26), rng.choice(['VISA', 'VISA / VISUM', 'ENTRY VISA']), 40, bold=True, fill=c.accent)
        c.text((330, 36), P['country'], 24)
        c.photo([44, 140, 300, 470])
        c.fields([('Valid for', 'ALL TERRITORY'), ('From', fmt(P['issue'], ds)), ('Until', fmt(P['expiry'], ds)), ('Type of visa', rng.choice(['C', 'D', 'TOURIST'])), ('Number of entries', rng.choice(['01', '02', 'MULT'])), ('Duration of stay', f"{rng.choice([30, 60, 90])} DAYS"),
                  ('Issued in', P['city'].upper()), ('Passport number', alnum(rng, 8, 2)), ('Surname, name', f"{P['surname']} {P['given']}".upper())], 340, 130, 300, 3, 14, 24, 100)
        lines = mrva(P['code'], num, P['code'], yymmdd(P['dob']), P['sex'], yymmdd(P['expiry']), P['surname'], P['given'])
        c.mrz(lines, 700, 37)
        truth['mrz'] = lines
    else:  # e-visa or entry permit on a sheet of paper
        c = Card(rng, 1100, 1400, tint=(248, 248, 244))
        c.text((50, 40), rng.choice(['UNITED ARAB EMIRATES', P['country']]), 32, bold=True, fill=c.accent)
        c.text((50, 90), rng.choice(['Entry Permit / e-Visa', 'ELECTRONIC VISA', 'Residence Permit', 'ENTRY PERMIT']), 24, fill=(170, 40, 40))
        c.fields([('Entry Permit No.', f"{digits(rng, 3)}/{P['issue'][0]}/{digits(rng, 7)}"), ('Visa Type', rng.choice(['TOURIST SINGLE ENTRY', 'VISIT VISA', 'EMPLOYMENT'])), ('Full Name', f"{P['given']} {P['surname']}".upper()), ('Nationality', P['nat']),
                  ('Passport No.', alnum(rng, 8, 2)), ('Date of Birth', fmt(P['dob'], ds)), ('Date of Issue', fmt(P['issue'], ds)), ('Valid Until', fmt(P['expiry'], ds)), ('Duration of Stay', f"{rng.choice([30, 60, 90])} DAYS"),
                  ('Sponsor', rng.choice(OWNERS)), ('UID No.', digits(rng, 9))], 60, 180, 900, 1, 18, 28, 100)
    c.specimen()
    return c.img, truth


MAKERS = {'passport': make_passport, 'id_card': make_id_card, 'emirates_id': make_emirates_id, 'driving_licence': make_driving_licence, 'vehicle_licence': make_vehicle_licence, 'visa': make_visa}


# ---------- how it was photographed ----------
def photograph(img, rng):
    """Return the image as a person might send it, plus a note on what was done."""
    mode = rng.choices(['flat', 'desk', 'sideways', 'glare', 'blur'], weights=[38, 26, 12, 12, 12])[0]
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    if mode in ('desk', 'sideways', 'glare') or rng.random() < 0.25:
        # lay it on a surface and shoot at a slight angle
        W, H = int(w * rng.uniform(1.15, 1.5)), int(h * rng.uniform(1.15, 1.6))
        base = np.array(rng.choice([(40, 38, 36), (70, 62, 54), (28, 30, 40), (90, 86, 80), (52, 44, 40), (20, 20, 22)]), np.float32)
        bg = np.clip(base + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 6, (H, W, 3)), 0, 255).astype(np.float32)
        j = lambda: rng.uniform(-0.035, 0.035)
        ox, oy = (W - w) / 2, (H - h) / 2
        dst = np.float32([[ox + w * j(), oy + h * j()], [ox + w * (1 + j()), oy + h * j()], [ox + w * (1 + j()), oy + h * (1 + j())], [ox + w * j(), oy + h * (1 + j())]])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        warped = cv2.warpPerspective(a, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H))[..., None]
        a = warped * mask + bg * (1 - mask)
        gx = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.85, 1.05), W)[None, :, None]
        a = a * gx  # uneven light across the scene
    if mode == 'glare':
        H, W = a.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy, r = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H, rng.uniform(0.12, 0.22) * W
        g = np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.6)) ** 2))[..., None]
        a = a + g * rng.uniform(90, 150)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if mode == 'blur' or rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.7, 1.9) if mode == 'blur' else rng.uniform(0.3, 0.8)))
    turned = 0
    if mode == 'sideways':
        turned = rng.choice([90, 270, 180])
        out = out.rotate(turned, expand=True)
    k = rng.uniform(0.55, 1.0)  # people send all sorts of sizes
    out = out.resize((max(480, int(out.width * k)), max(300, int(out.height * k * out.width / out.width))), Image.LANCZOS) if k < 0.97 else out
    return out, mode, turned, rng.randint(55, 90)


def build(split, per_type, countries, seed):
    rng = random.Random(seed)
    truth = []
    for t, make in MAKERS.items():
        d = os.path.join(OUT, split, t)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        for i in range(per_type):
            P = person(rng, countries)
            img, fields = make(rng, P)
            shot, mode, turned, q = photograph(img, rng)
            name = f'{t}-{i + 1:03d}-{mode}.jpg'
            shot.convert('RGB').save(os.path.join(d, name), 'JPEG', quality=q)
            truth.append({'file': f'{t}/{name}', 'type': t, 'country': P['code'], 'shot': mode, 'turned': turned, 'fields': {k: v for k, v in fields.items() if k != 'mrz'}, 'mrz': fields.get('mrz')})
    with open(os.path.join(OUT, split, 'truth.json'), 'w', encoding='utf8', newline='\n') as f:
        json.dump({'about': 'Made-up specimen documents. No real person, number or document appears.', 'documents': truth}, f, indent=1)
    by = {}
    for x in truth:
        by[x['shot']] = by.get(x['shot'], 0) + 1
    print(split, len(truth), 'documents', by, 'with code lines', sum(1 for x in truth if x['mrz']))


if __name__ == '__main__':
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    build('train', n_train, COUNTRIES[:-TEST_ONLY], 20261119)
    # the test set mixes countries seen in training with six that never were
    build('test', n_test, COUNTRIES, 77001)
    # a third set that nobody looks at while the reading rules are being improved: the number to report
    build('final', n_test, COUNTRIES, 990031)
