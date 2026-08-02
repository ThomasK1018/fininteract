"""Take numbers out of math mode in prose/captions and replace in-text math
symbols with natural language. Tabular bodies and displayed equations are left
alone. Genuine variables and formulas are kept in math."""
import io, os, re, sys, glob

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.chdir(r'C:\Users\thoma\Downloads\finteractcomp\finteractcomp\paper')
BS = chr(92)
APPLY = '--apply' in sys.argv

FILES = sorted(glob.glob('sections/*.tex'))

# Regions we do not touch: tabular bodies, displayed equations.
MASKS = [
    re.compile(re.escape(BS) + r'begin\{tabular\}.*?' + re.escape(BS) + r'end\{tabular\}', re.S),
    re.compile(re.escape(BS) + r'\[.*?' + re.escape(BS) + r'\]', re.S),
]

# Bodies that stay in math mode: real variables and formulas.
KEEP = re.compile(
    r'^(?:'
    r'[A-Za-z]|[A-Za-z]_\{?\w+\}?|'
    r'\(Q, C, A\)|'
    + re.escape(BS) + r'(?:Delta|kappa|mathcal|log|pm1|surd).*|'
    r'A_\{?d\}?' + re.escape(BS) + r'neq A|'
    r'H_\{?0\}?\s*=.*|'
    r'.*' + re.escape(BS) + r'mathcal.*|'
    r'.*' + re.escape(BS) + r'log.*|'
    r'n_' + re.escape(BS) + r'text.*|'
    r'[A-Za-z]\{?[=<>]\}?[A-Za-z].*'
    r')$'
)

PURE = re.compile(r'^[\d\s.,%+\-]+$|^[\d\s.,]+' + re.escape(BS) + r'%$')

# Explicit, hand-checked replacements for every symbol case in the paper.
MAP = {
    r'\gg': 'far exceeds',
    r'\to': 'to',
    r'\rightarrow': 'to',
    r'\times': 'times',
    r'3\times': '3 times', r'2\times': '2 times', r'2.1\times': '2.1 times',
    r'2.3\times': '2.3 times', r'2.5\times': '2.5 times', r'3.1\times': '3.1 times',
    r'6.5\times': '6.5 times', r'4\times': '4 times',
    r'\approx': 'about', r'\approx\ ': 'about ',
    r'\approx0.9': 'about 0.9', r'\approx0\%': 'about 0\\%',
    r'\approx N/0.113': 'about $N/0.113$',
    r'\sim': 'about',
    r'{\sim}2': 'about 2', r'{\sim}50': 'about 50', r'{\sim}60': 'about 60',
    r'{\sim}0.9': 'about 0.9', r'{\sim}4': 'about 4', r'{\sim}79\%': 'about 79\\%',
    r'{\sim}0\%': 'about 0\\%', r'{\sim}2$ accumulated': 'about 2 accumulated',
    r'\geq 2': 'at least 2', r'\geq 1': 'at least 1', r'\geq1': 'at least 1',
    r'\leq0.16': 'at most 0.16', r'\leq1': 'at most 1', r'\leq 1': 'at most 1',
    r'\leq23\%': 'at most 23\\%', r'\leq9': 'at most 9',
    r'{<}1\%': 'below 1\\%', r'{<}0.05': 'below 0.05',
    r'\pm1\%': 'plus or minus 1\\%',
    r'53{:}120': '53:120',
    r'H_0{\approx}3.6': '$H_0$ about 3.6',
    r'H_0{=}1': '$H_0$ of 1',
    r'4\text{B}\!\to\!32\text{B}': '4B to 32B',
    r'.00\to.68': '.00 to .68', r'.00\to.58': '.00 to .58',
    r'.05\!\to\!.26': '.05 to .26', r'.12\!\to\!.89': '.12 to .89',
    r'0.63\!\to\!0.93': '0.63 to 0.93', r'1000\!\to\!1551': '1000 to 1551',
    r'=0': 'of 0', r'=1.0': 'of 1.0',
}
NVAR = re.compile(r'^([nNB])\{?=\}?([\d.,]+)$')     # $n{=}9$ -> n = 9
# ---- second pass additions ----
MAP.update({
    r'\leftrightarrow': '-',
    r'=0.125': 'of 0.125',
    r'\sim0.77': 'about 0.77',
    r'\approx0.05': 'about 0.05',
    r'[14.5,26.6]': '[14.5, 26.6]',
    r'0.6/0.0/0.0\%': '0.6/0.0/0.0\\%',
    r'0.0/0.0\%': '0.0/0.0\\%',
    r'34/28/26\%': '34/28/26\\%',
    r'30/1{,}847': '30/1,847',
    r'n\leq1': '$n$ at most 1',
    r'n{\le}9': '$n$ at most 9',
    r'\approx\$54': 'about \\$54',
    r'\approx\$155': 'about \\$155',
    r'\approx\$309': 'about \\$309',
})
# numeric transitions:  $12.1\to4.6$ -> 12.1 to 4.6   (also \! spacing form)
ARROW = re.compile(r'^([\d.,]+\\?%?)\\(?:!)?\\?to\\?(?:!)?([\d.,]+\\?%?)$')
NUMTO = re.compile(r'^([\d.,]+(?:\\%)?)\\to([\d.,]+(?:\\%)?)$')


unhandled = {}
changed_total = 0

for f in FILES:
    src = io.open(f, encoding='utf-8').read()
    spans = []
    for rx in MASKS:
        spans += [(m.start(), m.end()) for m in rx.finditer(src)]

    def masked(i):
        return any(a <= i < b for a, b in spans)

    out, i, nchg = [], 0, 0
    while i < len(src):
        c = src[i]
        if c == BS and i + 1 < len(src) and src[i + 1] == '$':   # escaped \$ : dollar amount
            out.append(src[i:i + 2]); i += 2; continue
        if c == '$' and not masked(i):
            j = src.find('$', i + 1)
            while j != -1 and src[j - 1] == BS:
                j = src.find('$', j + 1)
            if j == -1:
                out.append(c); i += 1; continue
            body = src[i + 1:j]
            rep = None
            if body in MAP:
                rep = MAP[body]
            elif PURE.match(body):
                rep = body
            elif NVAR.match(body):
                g = NVAR.match(body)
                rep = '%s = %s' % (g.group(1), g.group(2))
            elif NUMTO.match(body):
                g = NUMTO.match(body)
                rep = '%s to %s' % (g.group(1), g.group(2))
            elif KEEP.match(body):
                rep = None
            else:
                unhandled.setdefault(body, 0)
                unhandled[body] += 1
            if rep is not None:
                out.append(rep); nchg += 1
            else:
                out.append(src[i:j + 1])
            i = j + 1
            continue
        out.append(c); i += 1

    new = ''.join(out)
    if nchg:
        changed_total += nchg
        print('%-28s %3d conversions' % (f, nchg))
        if APPLY:
            io.open(f, 'w', encoding='utf-8').write(new)

print('\nTOTAL conversions: %d' % changed_total)
if unhandled:
    print('\nUNHANDLED (left in math, review):')
    for b, c in sorted(unhandled.items(), key=lambda t: -t[1]):
        print('   %3dx  %s' % (c, b))
print('\n%s' % ('APPLIED' if APPLY else 'DRY RUN -- rerun with --apply'))
