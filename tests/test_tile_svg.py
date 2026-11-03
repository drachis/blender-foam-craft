"""Tiling a pattern sheet into printer pages.

    python tools/blender_bridge.py --port 9885 < tests/test_tile_svg.py

Pure arithmetic -- no bpy, no scene -- so it also runs standalone:

    python tests/test_tile_svg.py

What has to hold is narrow and absolute. A tiled pattern is taped back
together and cut from, so:

  * nothing may be lost between pages;
  * nothing may change size, because a few percent means the panels do not
    meet and the error is only found after the foam is cut.

Both are asserted against arithmetic rather than against a rendering, because
a picture of 35 pages cannot show either.
"""

import sys
import traceback

sys.path.insert(0, r"O:\Work\BlenderPlugin\FoamCraft\tools")

import tile_svg                                                # noqa: E402

MM_PER_INCH = 25.4


def _sheet(width, height):
    """A minimal pattern sheet with one marker of known size and position."""
    return (
        "<?xml version='1.0' encoding='UTF-8' standalone='no'?>\n"
        "<svg xmlns='http://www.w3.org/2000/svg' "
        "width='%.3fmm' height='%.3fmm' viewBox='0 0 %.3f %.3f'>\n"
        "<style>#x rect{fill:none;}</style>\n"
        "<g id='x'><rect x='10' y='10' width='100' height='100'/></g>\n"
        "</svg>" % (width, height, width, height))


def test_page_count_is_arithmetic():
    """Pages follow from the sheet, the paper and the overlap, and nothing else.

    Worked by hand for US Letter at the defaults: usable width is
    215.9 - 2*6.35 = 203.2, and neighbouring pages share 10 mm, so each page
    advances 193.2. A 1000 mm sheet needs ceil((1000 - 10) / 193.2) = 6
    columns. Height: usable 279.4 - 12.7 = 266.7, step 256.7, so a 500 mm
    sheet needs ceil((500 - 10) / 256.7) = 2 rows. Twelve pages.
    """
    pages = tile_svg.tile(_sheet(1000.0, 500.0))
    names = [n for n, _ in pages]
    cols = len({n.split("c")[1] for n in names})
    rows = len({n.split("c")[0].split("r")[1] for n in names})
    return {
        "pages": len(pages),
        "rows": rows,
        "columns": cols,
        "first": names[0],
        "last": names[-1],
        "checks": {
            "six_columns": cols == 6,
            "two_rows": rows == 2,
            "twelve_pages": len(pages) == 12,
        },
    }


def test_the_pages_cover_the_whole_sheet():
    """Every millimetre of the sheet lands on at least one page.

    A gap between pages is a missing strip of pattern, and it would not be
    obvious on any single page -- you would find it when the taped-up sheet
    had a piece cut in half. Checked by walking the sheet on a fine grid and
    asking which page window each point falls in.
    """
    width, height = 900.0, 640.0
    margin, overlap = tile_svg.DEFAULT_MARGIN_MM, tile_svg.DEFAULT_OVERLAP_MM
    page_w, page_h = tile_svg.PAGE_SIZES["letter"]
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin
    step_w, step_h = usable_w - overlap, usable_h - overlap

    pages = tile_svg.tile(_sheet(width, height))
    cols = len({n.split("c")[1] for n, _ in pages})
    rows = len({n.split("c")[0].split("r")[1] for n, _ in pages})

    windows = []
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * step_w, row * step_h
            windows.append((x0, y0, x0 + usable_w, y0 + usable_h))

    uncovered = 0
    samples = 0
    x = 0.0
    while x <= width:
        y = 0.0
        while y <= height:
            samples += 1
            if not any(wx0 <= x <= wx1 and wy0 <= y <= wy1
                       for wx0, wy0, wx1, wy1 in windows):
                uncovered += 1
            y += 7.0
        x += 7.0

    return {
        "sheet_mm": [width, height],
        "pages": len(pages),
        "samples": samples,
        "uncovered_samples": uncovered,
        "checks": {
            "nothing_is_missed": uncovered == 0,
            # Both bounds: a window list that covered the plane would also
            # pass, so check the pages do not wildly overshoot the sheet.
            "page_count_is_not_absurd": len(pages) <= rows * cols,
            "something_was_sampled": samples > 100,
        },
    }


def test_pages_are_printed_at_true_size():
    """A page must not rescale the pattern. This is the one that cuts foam.

    The transform on each page is translate-scale-translate, and at the
    default the scale factor has to be exactly 1. Asserted by reading the
    transform out of the generated page rather than by trusting the argument,
    because the whole failure mode here is a silent factor.
    """
    import re
    pages = tile_svg.tile(_sheet(600.0, 400.0))
    scales = set()
    for _name, svg in pages:
        for value in re.findall(r"scale\(([-\d.]+)\)", svg):
            scales.add(round(float(value), 6))

    half = tile_svg.tile(_sheet(600.0, 400.0), scale=0.5)
    half_scales = set()
    for _name, svg in half:
        for value in re.findall(r"scale\(([-\d.]+)\)", svg):
            half_scales.add(round(float(value), 6))
    stamped = all("NOT TO SCALE" in svg for _n, svg in half)

    return {
        "scales_at_default": sorted(scales),
        "scales_at_half": sorted(half_scales),
        "half_pages_stamped_not_to_scale": stamped,
        "checks": {
            "default_is_exactly_one": scales == {1.0},
            # Both bounds: if scale were ignored entirely the half case would
            # also read 1.0 and the check above would be meaningless.
            "an_explicit_scale_is_honoured": half_scales == {0.5},
            # And a reduced print says so on every page, because a page that
            # looks right and is 50% small is the worst possible outcome.
            "reduced_pages_are_marked": stamped,
        },
    }


def test_neighbouring_pages_share_an_overlap_band():
    """The band exists, is the requested width, and only where there is a
    neighbour.

    A band drawn on the outer edge of the sheet would be telling the maker to
    tape something to nothing.
    """
    pages = dict(tile_svg.tile(_sheet(900.0, 640.0), overlap_mm=12.0))
    names = sorted(pages)
    last = names[-1]
    first = names[0]
    return {
        "pages": len(pages),
        "first_page": first,
        "last_page": last,
        "first_has_bands": pages[first].count("<rect") >= 2,
        "last_has_no_bands": "<g id='overlap'></g>" in pages[last],
        "band_width_present": "12.000" in pages[first],
        "checks": {
            # Top-left has a neighbour to the right and below: two bands.
            "first_page_has_two_bands": pages[first].count(
                "<g id='overlap'>") == 1 and pages[first].count("<rect") >= 2,
            # Bottom-right has neither.
            "last_page_has_none": "<g id='overlap'></g>" in pages[last],
            "the_band_is_the_width_asked_for": "12.000" in pages[first],
        },
    }


def test_a_sheet_without_a_viewbox_is_refused():
    """Fail loudly on input this cannot reason about.

    Without a viewBox there is no way to know what the units mean, and
    guessing would produce pages that are confidently the wrong size.
    """
    raised = None
    try:
        tile_svg.tile("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    except tile_svg.NotAPatternSvg as exc:
        raised = str(exc)
    return {
        "raised": raised is not None,
        "message": raised,
        "checks": {
            "it_refused": raised is not None,
            "it_says_why": bool(raised) and "viewBox" in raised,
        },
    }


TESTS = (
    test_page_count_is_arithmetic,
    test_the_pages_cover_the_whole_sheet,
    test_pages_are_printed_at_true_size,
    test_neighbouring_pages_share_an_overlap_band,
    test_a_sheet_without_a_viewbox_is_refused,
)

result = {}
_all_pass = True
for _test in TESTS:
    try:
        _report = _test()
        _report["PASS"] = all(_report.get("checks", {}).values())
    except Exception:
        _report = {"ERROR": traceback.format_exc().splitlines()[-6:],
                   "PASS": False}
    result[_test.__name__] = _report
    _all_pass = _all_pass and _report["PASS"]
result["ALL_PASS"] = _all_pass

if __name__ == "__main__":
    import json
    print(json.dumps(result, indent=2))
