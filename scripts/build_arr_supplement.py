"""Build the two anonymized ARR supplement archives.

ARR gives a submission one Software slot and one Data slot (200 MB each), so the
single AAAI-style supplement is split in two:

    arr_software.zip -> scripts/, configs/ (example only), requirements.txt
    arr_data.zip     -> data/final/, data/results/

Every file is scanned for author-identifying strings and credentials before it is
written, and the build aborts if anything matches. Run from the repository root:

    python scripts/build_arr_supplement.py
"""
import io
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'submission', 'arr')
LIMIT_MB = 200

# --- what goes where -------------------------------------------------------
SOFTWARE_TREES = ['scripts']
SOFTWARE_FILES = [('requirements.txt', 'requirements.txt'),
                  (os.path.join('configs', 'openrouter.example.json'),
                   'configs/openrouter.example.json')]
DATA_TREES = [os.path.join('data', 'final'), os.path.join('data', 'results')]

# Never package these, whatever tree they turn up in.
EXCLUDE_NAMES = {
    'openrouter.json',            # holds a live API key
    'build_arr_supplement.py',    # submission tooling, not research code
    '.env', '.env.local', 'secrets.json', 'credentials.json',
    '.DS_Store', 'Thumbs.db',
}
EXCLUDE_DIRS = {'__pycache__', '.git', '.ipynb_checkpoints', '.venv'}
EXCLUDE_EXT = {'.pyc', '.pyo', '.log'}

# --- anonymity / credential gate -------------------------------------------
FORBIDDEN = {
    'author name':    r'(?i)\bkwok\b|thomask1018|simplew4y|tk1018',
    'institution':    r'(?i)\bucla\b|g\.ucla\.edu',
    'repo URL':       r'(?i)github\.com/[A-Za-z0-9_-]+',
    'home path':      r'(?i)[/\\]Users[/\\][A-Za-z0-9._-]+',
    'OpenAI key':     r'sk-[A-Za-z0-9]{20,}',
    'OpenRouter key': r'sk-or-[A-Za-z0-9-]{10,}',
    'Anthropic key':  r'sk-ant-[A-Za-z0-9-]{10,}',
    'HF token':       r'hf_[A-Za-z0-9]{20,}',
    'AWS key':        r'AKIA[0-9A-Z]{16}',
}
FORBIDDEN = {k: re.compile(v) for k, v in FORBIDDEN.items()}
BINARY_EXT = {'.png', '.jpg', '.pdf', '.zip', '.gz', '.npy', '.pt', '.bin'}


def scan(path, arcname, problems):
    if os.path.splitext(path)[1].lower() in BINARY_EXT:
        return
    try:
        txt = io.open(path, encoding='utf-8', errors='replace').read()
    except Exception:
        return
    for label, rx in FORBIDDEN.items():
        m = rx.search(txt)
        if m:
            ln = txt[:m.start()].count('\n') + 1
            problems.append('%s: %s at %s:%d (%r)'
                            % (label, arcname, path, ln, m.group(0)[:40]))


def collect(trees, files):
    out = []
    for tree in trees:
        for dp, dns, fns in os.walk(os.path.join(ROOT, tree)):
            dns[:] = [d for d in dns if d not in EXCLUDE_DIRS]
            for fn in sorted(fns):
                if fn in EXCLUDE_NAMES:
                    continue
                if os.path.splitext(fn)[1].lower() in EXCLUDE_EXT:
                    continue
                full = os.path.join(dp, fn)
                arc = os.path.relpath(full, ROOT).replace(os.sep, '/')
                out.append((full, arc))
    for rel, arc in files:
        full = os.path.join(ROOT, rel)
        if os.path.exists(full):
            out.append((full, arc))
        else:
            print('  WARNING: missing %s' % rel)
    return out


def build(zip_name, top, readme_src, members):
    problems = []
    for full, arc in members:
        scan(full, arc, problems)
    scan(os.path.join(OUT, readme_src), 'README.md', problems)
    if problems:
        print('\nABORTED: %s would leak identifying data or credentials:' % zip_name)
        for p in problems:
            print('   ' + p)
        return None

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, zip_name)
    raw = 0
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(os.path.join(OUT, readme_src), '%s/README.md' % top)
        for full, arc in members:
            z.write(full, '%s/%s' % (top, arc))
            raw += os.path.getsize(full)
    mb = os.path.getsize(dest) / 1048576.0
    print('%-20s %3d files  %6.1f MB raw -> %5.2f MB zipped  %s'
          % (zip_name, len(members) + 1, raw / 1048576.0, mb,
             'OK' if mb < LIMIT_MB else 'OVER 200 MB LIMIT'))
    return mb


def main():
    sw = collect(SOFTWARE_TREES, SOFTWARE_FILES)
    da = collect(DATA_TREES, [])
    a = build('arr_software.zip', 'fininteract_software', 'SOFTWARE_README.md', sw)
    b = build('arr_data.zip', 'fininteract_data', 'DATA_README.md', da)
    if a is None or b is None:
        sys.exit(1)
    print('\nwrote both archives to %s' % OUT)


if __name__ == '__main__':
    main()
