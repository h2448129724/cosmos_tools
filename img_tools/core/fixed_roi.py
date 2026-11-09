"""Fixed-size ROI placement for per-image crop adjustment.

Rectangles are half-open pixel boxes: the origin is inclusive and the far
edges are exclusive.  The functions only decide geometry.  Decoding, drawing
and file writes stay in the shell.
"""

from __future__ import annotations


Box = tuple[int, int, int, int]


def resize_fixed_box(
    box: Box,
    size: tuple[int, int],
    image_size: tuple[int, int],
) -> Box:
    """Resize around the current center, then keep the whole box inside the image.

    The requested size is reduced when it is larger than the image.  Integer
    rounding keeps the center when the new size is even; an odd size can land
    half a pixel away.
    """

    width, height = _fit_size(size, image_size)
    x1, y1, x2, y2 = _valid_box(box)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    origin_x = round(center_x - width / 2)
    origin_y = round(center_y - height / 2)
    return _place(origin_x, origin_y, width, height, image_size)


def inherit_fixed_box(box: Box, image_size: tuple[int, int]) -> Box:
    """Keep the previous origin and size, shifting inward only to stay inside."""

    image_width, image_height = _valid_image_size(image_size)
    x1, y1, x2, y2 = _valid_box(box)
    width = min(x2 - x1, image_width)
    height = min(y2 - y1, image_height)
    return _place(x1, y1, width, height, (image_width, image_height))


def resolve_fixed_boxes(
    *,
    saved: tuple[Box, ...] | None,
    previous: tuple[Box, ...] | None,
    image_size: tuple[int, int],
) -> tuple[Box, ...]:
    """Keep every remembered box, or copy all of the previous image's boxes.

    An empty saved list means the image was visited and cleared.  That does not
    inherit again.  Sizes stay as drawn; only the position is pulled inside.
    """

    source = saved if saved is not None else previous
    if not source:
        return ()
    return tuple(inherit_fixed_box(box, image_size) for box in source)


def resolve_fixed_box(
    *,
    saved: Box | None,
    previous: Box | None,
    size: tuple[int, int],
    image_size: tuple[int, int],
) -> Box | None:
    """Return the box for one image.

    A remembered box keeps its center and adopts the requested size.  An image
    without a remembered box inherits the previous origin, then adopts the same
    requested size.  Neither source produces a box until the user places one.
    """

    if saved is not None:
        return resize_fixed_box(saved, size, image_size)
    if previous is None:
        return None
    inherited = inherit_fixed_box(previous, image_size)
    return resize_fixed_box(inherited, size, image_size)


def _fit_size(size: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    image_width, image_height = _valid_image_size(image_size)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("框的宽高必须大于 0")
    return min(width, image_width), min(height, image_height)


def _valid_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片宽高必须大于 0")
    return image_width, image_height


def _valid_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("框必须是有效的半开区域")
    return x1, y1, x2, y2


def _place(
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    image_size: tuple[int, int],
) -> Box:
    image_width, image_height = image_size
    max_x = image_width - width
    max_y = image_height - height
    origin_x = min(max(0, origin_x), max_x)
    origin_y = min(max(0, origin_y), max_y)
    return origin_x, origin_y, origin_x + width, origin_y + height


__all__ = [
    "Box",
    "inherit_fixed_box",
    "resize_fixed_box",
    "resolve_fixed_box",
    "resolve_fixed_boxes",
]
