"""Slice a foam pattern SVG into printer-sized pages with registration marks.

    python tools/tile_svg.py pattern.svg
    python tools/tile_svg.py pattern.svg --out pages/ --overlap 12
    python tools/tile_svg.py pattern.svg --page a4
    python tools/tile_svg.py pattern.svg --scale 0.5     # half size test print

A FoamCraft sheet is as big as the pattern needs to be -- a 0.4 m sphere comes
out around 1.2 m square, and a horn can be larger. No home printer takes that,
so it has to be tiled.

WHAT IT PRESERVES

The exported sheet is in real millimetres: 100 mm in the file is 100 mm of
foam. Tiling must not quietly change that, or every piece comes out the wrong
size and the panels do not meet. So pages are cut from the sheet at 1:1 by
default, and `--scale` exists only to be explicit about wanting otherwise --
for a reduced proof print, say.

Each page carries the scale strip's logic in miniature: crop marks at the
corners, the page's row/column, and an overlap band shaded so there is a
visible region to align against its neighbour. Measure the printed ruler on
page 1 before trusting any of it.

HOW IT WORKS

The source SVG is not re-rendered or re-parsed into shapes. Its whole body is
embedded in each page inside a translated, clipped group, so whatever the
exporter drew -- fills, outlines, darts, text -- appears identically on the
pages. Nothing about the drawing is reinterpreted, which is the only way to be
sure a tiled print matches the sheet it came from.

This is importable: `tile(svg_text, ...)` returns a list of (filename, svg)
pairs without touching the filesystem, so a Blender panel can call it and
write the files itself.
"""

import argparse
import os
import re
import sys

MM_PER_INCH = 25.4

# width x height in millimetres, portrait.
PAGE_SIZES = {
    "letter": (8.5 * MM_PER_INCH, 11.0 * MM_PER_INCH),
    "legal": (8.5 * MM_PER_INCH, 14.0 * MM_PER_INCH),
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
}

DEFAULT_MARGIN_MM = 6.35      # a quarter inch; most printers cannot do less
DEFAULT_OVERLAP_MM = 10.0     # shared band between neighbouring pages
CROP_MM = 5.0
LABEL_FONT_MM = 3.5


class NotAPatternSvg(ValueError):
    """The input does not look like a FoamCraft sheet."""


def _parse(svg_text):
    """Pull the viewBox and the drawable body out of an SVG.

    Deliberately textual rather than a real XML parse: the body is copied
    through verbatim, so the less this understands about it the fewer ways it
    has to change it.
    """
    m = re.search(r"viewBox\s*=\s*['\"]\s*([-\d.]+)\s+([-\d.]+)\s+"
                  r"([\d.]+)\s+([\d.]+)\s*['\"]", svg_text)
    if not m:
        raise NotAPatternSvg(
            "no viewBox found. This expects an SVG exported by FoamCraft, "
            "whose viewBox is the sheet size in millimetres.")
    min_x, min_y, width, height = (float(m.group(i)) for i in range(1, 5))

    open_tag = re.search(r"<svg\b[^>]*>", svg_text)
    close_tag = svg_text.rindex("</svg>")
    if not open_tag:
        raise NotAPatternSvg("no <svg> element found.")
    body = svg_text[open_tag.end():close_tag]
    return (min_x, min_y, width, height), body


def _crop_marks(x0, y0, x1, y1):
    """Corner ticks, pointing outward, for trimming and aligning."""
    out = []
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        out.append(f'<line x1="{cx:.3f}" y1="{cy:.3f}" '
                   f'x2="{cx + dx * CROP_MM:.3f}" y2="{cy:.3f}"/>')
        out.append(f'<line x1="{cx:.3f}" y1="{cy:.3f}" '
                   f'x2="{cx:.3f}" y2="{cy + dy * CROP_MM:.3f}"/>')
    return out


def tile(svg_text, page="letter", margin_mm=DEFAULT_MARGIN_MM,
         overlap_mm=DEFAULT_OVERLAP_MM, scale=1.0, landscape=False,
         stem="page"):
    """Cut `svg_text` into pages. Returns [(filename, svg_text), ...].

    No file I/O, so a caller in Blender can decide where the pages go.
    """
    if page not in PAGE_SIZES:
        raise ValueError("unknown page size %r; known: %s"
                         % (page, ", ".join(sorted(PAGE_SIZES))))
    page_w, page_h = PAGE_SIZES[page]
    if landscape:
        page_w, page_h = page_h, page_w

    (min_x, min_y, sheet_w, sheet_h), body = _parse(svg_text)
    sheet_w *= scale
    sheet_h *= scale

    # What one page can actually carry, after margins.
    usable_w = page_w - 2 * margin_mm
    usable_h = page_h - 2 * margin_mm
    if usable_w <= overlap_mm or usable_h <= overlap_mm:
        raise ValueError("margins and overlap leave no usable page area")

    # Neighbouring pages share `overlap_mm`, so each page advances by less
    # than its own width. That shared band is what you align and tape.
    step_w = usable_w - overlap_mm
    step_h = usable_h - overlap_mm
    cols = max(1, int(-(-max(sheet_w - overlap_mm, 0.001) // step_w)))
    rows = max(1, int(-(-max(sheet_h - overlap_mm, 0.001) // step_h)))

    pages = []
    for row in range(rows):
        for col in range(cols):
            off_x = col * step_w
            off_y = row * step_h
            # Where this page's window sits in sheet coordinates, allowing
            # for the source viewBox origin and the requested scale.
            tx = margin_mm - off_x
            ty = margin_mm - off_y

            marks = _crop_marks(margin_mm, margin_mm,
                                page_w - margin_mm, page_h - margin_mm)

            # The overlap band, shaded. Drawn only on the edges that actually
            # have a neighbour, so the outer border of the whole sheet is not
            # covered in bands that mean nothing.
            bands = []
            if col < cols - 1:
                bands.append(
                    f'<rect x="{page_w - margin_mm - overlap_mm:.3f}" '
                    f'y="{margin_mm:.3f}" width="{overlap_mm:.3f}" '
                    f'height="{usable_h:.3f}"/>')
            if row < rows - 1:
                bands.append(
                    f'<rect x="{margin_mm:.3f}" '
                    f'y="{page_h - margin_mm - overlap_mm:.3f}" '
                    f'width="{usable_w:.3f}" height="{overlap_mm:.3f}"/>')

            label = (f"row {row + 1} of {rows}, column {col + 1} of {cols}"
                     f"  |  tape along the shaded band")
            if scale != 1.0:
                label += f"  |  PRINTED AT {scale * 100:.0f}% - NOT TO SCALE"

            page_svg = "\n".join([
                "<?xml version='1.0' encoding='UTF-8' standalone='no'?>",
                f"<svg xmlns='http://www.w3.org/2000/svg' "
                f"width='{page_w:.3f}mm' height='{page_h:.3f}mm' "
                f"viewBox='0 0 {page_w:.3f} {page_h:.3f}'>",
                "<style>"
                "#crop line{stroke:#000;stroke-width:0.25;}"
                "#overlap rect{fill:#000;fill-opacity:0.06;stroke:#888;"
                "stroke-width:0.2;stroke-dasharray:2 2;}"
                "#pagelabel text{font-family:sans-serif;fill:#000;}"
                "</style>",
                "<defs>",
                f"<clipPath id='win'><rect x='{margin_mm:.3f}' "
                f"y='{margin_mm:.3f}' width='{usable_w:.3f}' "
                f"height='{usable_h:.3f}'/></clipPath>",
                "</defs>",
                "<g clip-path='url(#win)'>",
                # Translate first, then scale, then shift away the source
                # origin. The body is copied through untouched.
                f"<g transform='translate({tx:.4f},{ty:.4f}) "
                f"scale({scale:.6f}) translate({-min_x:.4f},{-min_y:.4f})'>",
                body,
                "</g>",
                "</g>",
                f"<g id='overlap'>{''.join(bands)}</g>",
                f"<g id='crop'>{''.join(marks)}</g>",
                f"<g id='pagelabel'><text x='{margin_mm:.3f}' "
                f"y='{page_h - margin_mm / 2:.3f}' "
                f"font-size='{LABEL_FONT_MM}'>{label}</text></g>",
                "</svg>",
            ])
            pages.append(("%s_r%02dc%02d.svg" % (stem, row + 1, col + 1),
                          page_svg))
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", help="pattern SVG exported by FoamCraft")
    parser.add_argument("--out", default=None,
                        help="output directory (default: alongside the input)")
    parser.add_argument("--page", default="letter",
                        choices=sorted(PAGE_SIZES), help="paper size")
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_MM,
                        help="unprintable margin in mm (default 6.35 = 1/4in)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_MM,
                        help="shared band between pages, in mm, for taping")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="1.0 keeps real size. Anything else makes the "
                             "pages unusable for cutting; each page is "
                             "stamped NOT TO SCALE when you do")
    args = parser.parse_args()

    with open(args.svg, encoding="utf-8") as f:
        text = f.read()

    try:
        pages = tile(text, page=args.page, margin_mm=args.margin,
                     overlap_mm=args.overlap, scale=args.scale,
                     landscape=args.landscape,
                     stem=os.path.splitext(os.path.basename(args.svg))[0])
    except (NotAPatternSvg, ValueError) as exc:
        print("tile_svg: %s" % exc, file=sys.stderr)
        return 2

    out_dir = args.out or os.path.dirname(os.path.abspath(args.svg))
    os.makedirs(out_dir, exist_ok=True)
    for name, svg in pages:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(svg)

    (min_x, min_y, w, h), _ = _parse(text)
    print("sheet   : %.1f x %.1f mm" % (w * args.scale, h * args.scale))
    print("page    : %s%s" % (args.page, " landscape" if args.landscape else ""))
    print("pages   : %d, written to %s" % (len(pages), out_dir))
    if args.scale == 1.0:
        print("Measure the ruler on the pattern after printing. If the 100 mm "
              "bar is not 100 mm, the printer scaled it and the pieces will "
              "not fit.")
    else:
        print("NOT TO SCALE at %.0f%% -- proof print only." % (args.scale * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
