"""Draw an original apartment-style input with furniture and drafting distractions.

This is a program-generated interface example, not a real drawing or a dataset.
No wall truth is exported: evaluate a model's prediction separately from this image.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER = "#fffefb"
FLOOR = "#ededE8"
WALL = "#29333d"
DETAIL = "#68767b"
PALE = "#dce1db"
GLASS = "#7993a1"
FURNITURE = "#d0d9d3"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class _Draft:
    """Small drawing vocabulary with separate floor, detail, and barrier layers."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (1200, 900), PAPER)
        self.draw = ImageDraw.Draw(self.image)

    def line(self, points: list[tuple[int, int]], fill: str = DETAIL, width: int = 2) -> None:
        self.draw.line(points, fill=fill, width=width)

    def rect(self, box: tuple[int, int, int, int], fill: str, width: int = 2) -> None:
        self.draw.rectangle(box, fill=fill, outline=DETAIL, width=width)

    def text(self, xy: tuple[int, int], text: str, size: int = 16, fill: str = DETAIL) -> None:
        self.draw.text(xy, text, font=_font(size), fill=fill, anchor="mm")

    def bed(self, x: int, y: int, width: int = 105, height: int = 125) -> None:
        self.rect((x - 5, y - 7, x + width + 5, y + 12), "#b9c8c0")
        self.draw.rounded_rectangle(
            (x, y, x + width, y + height), radius=6, fill="#f9f9f3", outline=DETAIL, width=2
        )
        self.rect((x + 7, y + 8, x + width // 2 - 4, y + 30), "#e4e8df")
        self.rect((x + width // 2 + 4, y + 8, x + width - 7, y + 30), "#e4e8df")
        self.rect((x + 1, y + 44, x + width - 1, y + height - 1), "#d7ddd5")
        self.line([(x + 8, y + 55), (x + width - 8, y + 55)], PALE)
        self.rect((x - 30, y + 3, x - 10, y + 30), "#e0d9cc")
        self.rect((x + width + 10, y + 3, x + width + 30, y + 30), "#e0d9cc")

    def wardrobe(self, box: tuple[int, int, int, int], divisions: int) -> None:
        self.rect(box, "#dcdcd5")
        x1, y1, x2, y2 = box
        for index in range(1, divisions):
            x = x1 + (x2 - x1) * index // divisions
            self.line([(x, y1), (x, y2)], DETAIL, 1)
        for index in range(divisions):
            x = x1 + (x2 - x1) * (2 * index + 1) // (2 * divisions)
            self.line([(x - 4, (y1 + y2) // 2), (x + 4, (y1 + y2) // 2)], DETAIL, 1)

    def plant(self, x: int, y: int, radius: int = 17) -> None:
        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#d9dcc8",
            outline=DETAIL,
            width=2,
        )
        for dx, dy in ((-8, -4), (7, -6), (0, 8), (-4, 1)):
            self.draw.ellipse(
                (x + dx - 7, y + dy - 6, x + dx + 7, y + dy + 6),
                fill="#9bb69b",
                outline="#668d73",
                width=1,
            )

    def horizontal_window(self, x1: int, x2: int, y: int) -> None:
        self.line([(x1, y), (x2, y)], PAPER, 18)
        for offset in (-5, 0, 5):
            self.line([(x1, y + offset), (x2, y + offset)], GLASS, 2)
        for x in (x1, (x1 + x2) // 2, x2):
            self.line([(x, y - 8), (x, y + 8)], WALL, 2)

    def vertical_window(self, x: int, y1: int, y2: int) -> None:
        self.line([(x, y1), (x, y2)], PAPER, 18)
        for offset in (-5, 0, 5):
            self.line([(x + offset, y1), (x + offset, y2)], GLASS, 2)
        for y in (y1, (y1 + y2) // 2, y2):
            self.line([(x - 8, y), (x + 8, y)], WALL, 2)

    def door_up(self, x: int, y: int, width: int = 60) -> None:
        """A horizontal opening with its leaf swung into the room above."""
        self.line([(x + 2, y), (x + width - 2, y)], FLOOR, 13)
        self.line([(x, y), (x, y - width)], WALL, 3)
        self.draw.arc((x - width, y - width, x + width, y + width), 270, 360, fill=DETAIL, width=2)

    def door_left(self, x: int, y: int, width: int = 60, exterior: bool = False) -> None:
        """A vertical opening with its leaf swung left from the lower hinge."""
        self.line([(x, y - width + 2), (x, y - 2)], FLOOR, 19 if exterior else 13)
        self.line([(x, y), (x - width, y)], WALL, 3)
        self.draw.arc((x - width, y - width, x + width, y + width), 180, 270, fill=DETAIL, width=2)

    def dimension(self, start: tuple[int, int], end: tuple[int, int], label: str) -> None:
        self.line([start, end], DETAIL, 1)
        for x, y in (start, end):
            self.line([(x - 5, y + 5), (x + 5, y - 5)], DETAIL, 2)
        if start[1] == end[1]:
            self.text(((start[0] + end[0]) // 2, start[1] - 12), label, 14)
        else:
            self.text((start[0] + 27, (start[1] + end[1]) // 2), label, 14)


def draw_complex_demo() -> np.ndarray:
    """Return an original 1200×900 furnished apartment drawing as RGB uint8.

    The irregular outline, door swings, glazed openings, furniture, and drafting
    marks deliberately make this a richer input than the package's small demo.
    Dimension labels are illustrative. This function exports no prediction or
    wall mask and carries no claim about a model's recognition quality.
    """
    draft = _Draft()
    draw = draft.draw
    outline = [
        (150, 170),
        (800, 170),
        (800, 250),
        (1040, 250),
        (1040, 700),
        (790, 700),
        (790, 790),
        (320, 790),
        (320, 735),
        (150, 735),
    ]
    draw.polygon(outline, fill=FLOOR)

    # Surface marks stay low contrast; they are visual clutter, not wall truth.
    for box in ((158, 178, 352, 362), (368, 178, 557, 362), (158, 378, 352, 577)):
        for y in range(box[1] + 6, box[3], 14):
            draft.line([(box[0], y), (box[2], y)], "#dfe2dc", 1)
    for x in range(810, 1035, 24):
        draft.line([(x, 258), (x, 402)], "#d6dedb", 1)
    for y in range(258, 405, 24):
        draft.line([(808, y), (1032, y)], "#d6dedb", 1)
    for y in range(596, 730, 12):
        draft.line([(158, y), (311, y)], "#d5dbd2", 1)

    # Bedrooms: beds, lockers, a desk, and chairs all add non-wall ink.
    draft.bed(190, 220, 90, 118)
    draft.wardrobe((170, 185, 340, 205), 4)
    draft.bed(399, 213, 95, 118)
    draft.wardrobe((376, 183, 548, 203), 4)
    draft.bed(193, 401, 92, 122)
    draft.wardrobe((302, 390, 342, 470), 2)
    draft.rect((168, 543, 270, 567), "#ddd8cc")
    draw.rounded_rectangle((208, 520, 239, 540), radius=6, fill=FURNITURE, outline=DETAIL)
    draft.rect((179, 547, 202, 559), "#e9ece6", 1)
    draft.text((263, 352), "BEDROOM 1", 15)
    draft.text((470, 350), "BEDROOM 2", 15)
    draft.text((252, 384), "BEDROOM 3", 12)

    # Kitchen counter with cooker, sink, preparation board, and a refrigerator.
    draft.rect((578, 183, 786, 224), "#d9dfda")
    draft.rect((750, 224, 786, 335), "#d9dfda")
    draft.rect((583, 283, 628, 337), "#d8ddd8")
    draft.line([(583, 300), (628, 300)], DETAIL, 1)
    draft.rect((601, 190, 645, 216), "#eff3ed")
    draft.rect((608, 194, 639, 212), "#bfcfd0", 1)
    draw.arc((617, 185, 633, 201), 185, 350, fill=DETAIL, width=2)
    draft.rect((668, 188, 725, 217), "#e9ebe7")
    for x in (680, 710):
        for y in (197, 209):
            draw.ellipse((x - 6, y - 5, x + 6, y + 5), outline=DETAIL, width=2)
    draft.rect((755, 241, 780, 274), "#e7dbc2", 1)
    draft.text((688, 266), "KITCHEN", 17)
    draft.text((610, 385), "HALL", 14)

    # Bathroom: shower tray, toilet tank/bowl, vanity, and towels.
    draft.rect((957, 269, 1024, 337), "#d9e4e1")
    draft.line([(962, 274), (1019, 332)], GLASS, 1)
    draft.line([(1019, 274), (962, 332)], GLASS, 1)
    draw.ellipse((980, 292, 1001, 314), outline=DETAIL, width=1)
    draft.rect((816, 272, 877, 310), "#cfdad5")
    draw.ellipse((827, 279, 865, 304), fill="#eff1eb", outline=DETAIL, width=2)
    draft.rect((894, 269, 928, 284), "#f5f5ee")
    draw.ellipse((896, 280, 927, 318), fill="#f5f5ee", outline=DETAIL, width=2)
    draw.ellipse((902, 290, 921, 308), outline=DETAIL, width=1)
    draft.line([(981, 367), (1021, 367)], DETAIL, 3)
    draft.rect((990, 363, 1006, 381), "#bbd1c7", 1)
    draft.text((925, 353), "BATHROOM", 16)

    # Living room: vertical sofa, cushions, a rug, oval coffee table, and TV unit.
    draw.rounded_rectangle((393, 473, 724, 741), radius=10, fill="#e3e5de")
    draw.rounded_rectangle((397, 488, 480, 704), radius=10, fill=FURNITURE, outline=DETAIL, width=2)
    draft.rect((402, 494, 420, 698), "#bccbc1")
    for y in (500, 559, 618):
        draw.rounded_rectangle((425, y, 475, y + 51), radius=5, outline=DETAIL, width=1)
    for x, y in ((434, 508), (434, 660)):
        draft.rect((x, y, x + 25, y + 24), "#e5dfd1", 1)
    draw.ellipse((532, 559, 655, 630), fill="#d4c6ae", outline=DETAIL, width=2)
    draft.rect((558, 577, 585, 600), "#eef0e6", 1)
    draw.ellipse((609, 589, 625, 605), fill="#c7d7cf", outline=DETAIL)
    draw.rounded_rectangle((638, 461, 703, 518), radius=10, fill="#d4dacc", outline=DETAIL)
    draft.rect((745, 516, 774, 681), "#c8cfc9")
    draft.rect((755, 538, 767, 650), "#56666d")
    draft.text((574, 435), "LIVING ROOM", 20)
    draft.plant(749, 745, 20)

    # Dining table and six chairs; entry console and footwear bench.
    for y in (457, 494, 531):
        draw.rounded_rectangle((844, y, 871, y + 26), radius=5, fill=FURNITURE, outline=DETAIL)
        draw.rounded_rectangle((981, y, 1008, y + 26), radius=5, fill=FURNITURE, outline=DETAIL)
    draw.rounded_rectangle((873, 450, 978, 565), radius=12, fill="#d8ccb8", outline=DETAIL, width=2)
    for x in (891, 961):
        for y in (470, 508, 546):
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=DETAIL, width=1)
    draft.plant(925, 507, 10)
    draft.text((923, 427), "DINING", 17)
    draft.rect((828, 651, 935, 682), "#d4d7cc")
    draft.line([(855, 653), (855, 680)], DETAIL, 1)
    draft.line([(908, 653), (908, 680)], DETAIL, 1)
    draft.text((920, 615), "ENTRY", 16)
    draft.plant(1005, 597)

    # A private bedroom balcony, including a small table and planters.
    draw.ellipse((210, 632, 255, 677), fill="#e4daca", outline=DETAIL, width=2)
    draw.rounded_rectangle((185, 640, 204, 665), radius=4, outline=DETAIL, fill=FURNITURE)
    draw.rounded_rectangle((261, 640, 280, 665), radius=4, outline=DETAIL, fill=FURNITURE)
    draft.plant(179, 701)
    draft.plant(291, 698)
    draft.text((238, 613), "BALCONY", 15)

    # Barriers are a drawing layer only, never returned as evaluation truth.
    draft.line(outline + [outline[0]], WALL, 16)
    for points in (
        [(360, 170), (360, 585)],
        [(565, 170), (565, 370)],
        [(150, 370), (565, 370)],
        [(565, 355), (800, 355)],
        [(150, 585), (360, 585)],
        [(800, 250), (800, 700)],
        [(800, 410), (1040, 410)],
    ):
        draft.line(points, WALL, 10)
    draft.door_up(285, 370, 58)
    draft.door_up(505, 370, 50)
    draft.door_up(637, 355, 70)
    draft.door_up(850, 410, 62)
    draft.door_left(360, 555, 62)
    draft.door_left(1040, 680, 65, exterior=True)
    draft.line([(800, 461), (800, 553)], FLOOR, 13)
    draft.line([(800, 618), (800, 679)], FLOOR, 13)
    draft.horizontal_window(184, 317, 170)
    draft.horizontal_window(390, 533, 170)
    draft.horizontal_window(612, 769, 170)
    draft.horizontal_window(500, 722, 790)
    draft.vertical_window(150, 416, 522)
    draft.vertical_window(150, 620, 718)
    draft.vertical_window(1040, 282, 373)
    draft.vertical_window(1040, 444, 560)
    draft.horizontal_window(183, 302, 585)

    # Dimension strings and leader marks mimic a drafted floor-plan input.
    for x, y in ((150, 170), (360, 170), (565, 170), (800, 170), (1040, 250)):
        draft.line([(x, y - 14), (x, 106)], PALE, 1)
    draft.dimension((150, 113), (800, 113), "9750")
    draft.dimension((800, 113), (1040, 113), "3600")
    for x1, x2, label in ((150, 360, "3150"), (360, 565, "3075"), (565, 800, "3525")):
        draft.dimension((x1, 143), (x2, 143), label)
    for y in (250, 410, 700):
        draft.line([(1053, y), (1135, y)], PALE, 1)
    draft.dimension((1100, 250), (1100, 410), "2400")
    draft.dimension((1100, 410), (1100, 700), "4350")
    for x, y in ((320, 790), (790, 790)):
        draft.line([(x, y + 12), (x, 849)], PALE, 1)
    draft.dimension((320, 837), (790, 837), "7050")
    draft.line([(952, 733), (987, 733), (997, 707)], DETAIL, 1)
    draft.text((932, 737), "D-01", 12)
    draft.text((297, 43), "APARTMENT / SCHEMATIC PLAN", 24, WALL)
    draft.text((294, 70), "Original code-generated layout", 14)
    draft.text((959, 52), "REV. A   |   DEMONSTRATION", 13)
    draft.text((600, 878), "ILLUSTRATIVE DIMENSIONS  |  GENERATED EXAMPLE, NOT A REAL PROJECT", 12)
    return np.asarray(draft.image, dtype=np.uint8).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination PNG; keep output outside public source",
    )
    args = parser.parse_args()
    image = draw_complex_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"Generated example saved: {args.output} ({image.shape[1]}x{image.shape[0]}, RGB)")


if __name__ == "__main__":
    main()
