#!/usr/bin/env bash
# Render a Markdown file to PDF for people who want a document rather than a repo.
#
#   scripts/md2pdf.sh docs/CHECKPOINT1.zh.md            # -> docs/CHECKPOINT1.zh.pdf
#   scripts/md2pdf.sh docs/CHECKPOINT1.zh.md out.pdf
#
# markdown-it renders the body, Chromium prints it. Chromium here is a snap, so
# its /tmp is private to the sandbox -- the intermediate HTML has to live beside
# the source, which also makes relative image paths resolve without a <base>.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.md> [output.pdf]" >&2
    exit 2
fi

SRC="$(realpath "$1")"
[[ -f "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }
OUT="$(realpath -m "${2:-${SRC%.md}.pdf}")"
DIR="$(dirname "$SRC")"
TITLE="$(basename "${SRC%.md}")"

CHROME="$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)"
[[ -n "$CHROME" ]] || { echo "need chromium or google-chrome on PATH" >&2; exit 1; }

HTML="$DIR/.md2pdf-$$.html"
trap 'rm -f "$HTML"' EXIT

{
    cat <<'HEAD'
<!doctype html>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body {
    font-family: "Noto Sans CJK SC", "Noto Sans CJK TC", "Source Han Sans SC", sans-serif;
    font-size: 10.5pt; line-height: 1.7; color: #1a1a1a; margin: 0;
  }
  h1 { font-size: 19pt; margin: 0 0 .6em; padding-bottom: .3em; border-bottom: 2px solid #333; }
  h2 { font-size: 14pt; margin: 1.6em 0 .5em; padding-bottom: .2em; border-bottom: 1px solid #bbb; }
  h3 { font-size: 11.5pt; margin: 1.2em 0 .4em; }
  h1, h2, h3, h4 { break-after: avoid; }
  p, li { orphans: 2; widows: 2; }
  table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 9.5pt; }
  th, td { border: 1px solid #c8c8c8; padding: 5px 8px; text-align: left; vertical-align: top; }
  th { background: #f0f0f0; font-weight: 600; }
  table, pre, blockquote, img { break-inside: avoid; }
  code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f2f2f2;
         padding: 1px 4px; border-radius: 3px; }
  pre { background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 4px;
        padding: 9px 12px; overflow-wrap: break-word; white-space: pre-wrap; }
  pre code { background: none; padding: 0; }
  blockquote { margin: .8em 0; padding: .1em 1em; border-left: 3px solid #ccc; color: #555; }
  img { max-width: 100%; }
  a { color: #0b5cad; text-decoration: none; }
  hr { border: none; border-top: 1px solid #ddd; margin: 1.6em 0; }
</style>
HEAD
    printf '<title>%s</title>\n' "$TITLE"
    npx -y --quiet markdown-it "$SRC"
} > "$HTML"

"$CHROME" --headless --disable-gpu --no-sandbox \
    --no-pdf-header-footer --virtual-time-budget=10000 \
    --print-to-pdf="$OUT" "file://$HTML" 2>/dev/null

echo "$OUT"
