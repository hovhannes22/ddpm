import os
import argparse
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
FONT_SIZE = 200            # font size for all letters
IMG_W = 96    # output image dimensions
IMG_H = IMG_W
PAD_X, PAD_Y = 5, 5      # horizontal & vertical padding
# ------------------------------------------------------------------

# Full modern Armenian alphabet (38 letters), upper + lower case
UPPER = list("ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ")
LOWER = list("աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ")
LETTERS = UPPER + LOWER

def _fits_all_letters(font_path: str, size: int, max_w: int, max_h: int) -> bool:
    """
    Returns True if *every* LETTERS glyph, when rendered at this `size`,
    has a bounding box <= (max_w, max_h).
    """
    font = ImageFont.truetype(font_path, size)
    # dummy image just to measure
    img = Image.new("RGB", (max_w, max_h))
    draw = ImageDraw.Draw(img)

    for letter in LETTERS:
        bbox = draw.textbbox((0, 0), letter, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > max_w or h > max_h:
            return False
    return True

def find_max_font_size(font_path: str, max_w: int, max_h: int) -> int:
    """
    Binary‐search the largest integer font‐size so that all letters fit within
    (max_w, max_h). 
    """
    lo, hi = 1, max(max_w, max_h)  # hi could be content height or width
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits_all_letters(font_path, mid, max_w, max_h):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def make_images(fonts_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # compute the content area inside padding:
    content_w = IMG_W - 2 * PAD_X
    content_h = IMG_H - 2 * PAD_Y

    # gather all .ttf/.otf fonts
    font_files = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.lower().endswith((".ttf", ".otf"))
    ]
    if not font_files:
        raise RuntimeError(f"No .ttf/.otf files found in '{fonts_dir}'")

    total = 0
    for font_id, font_path in enumerate(font_files, start=1):
        # find the largest font size that fits
        optimal_size = find_max_font_size(font_path, content_w, content_h)
        font = ImageFont.truetype(font_path, optimal_size)
        print(f"[Font {font_id}/{len(font_files)}] "
              f"{os.path.basename(font_path)} → size {optimal_size}")

        for letter in LETTERS:
            img = Image.new("RGB", (IMG_W, IMG_H), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # exact bounding‐box of glyph at this size
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # center inside padded content area
            x = PAD_X + (content_w - text_w) / 2 - bbox[0]
            y = PAD_Y + (content_h - text_h) / 2 - bbox[1]

            draw.text((x, y), letter, font=font, fill=(0, 0, 0))
            case = 'lower' if letter in LOWER else 'upper'

            fname = f"{case}_{letter}_{font_id}.png"
            img.save(os.path.join(out_dir, fname))
            total += 1

    print(f"Done! Generated {total} images in '{out_dir}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Render each Armenian letter in every font under ./fonts/"
    )
    p.add_argument(
        "--output_dir",
        default="fonts",
        help="Directory where letter images will be saved (will be created if needed)",
    )
    p.add_argument(
        "--fonts-dir",
        default="input",
        help="Folder containing .ttf/.otf font files (default: ./fonts)",
    )
    args = p.parse_args()

    make_images(args.fonts_dir, args.output_dir)
