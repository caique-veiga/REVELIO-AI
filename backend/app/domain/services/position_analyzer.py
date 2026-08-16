from app.domain.entities.bounding_box import BoundingBox
from app.domain.entities.position import HorizontalPosition, Position, Region, VerticalPosition

_REGION_VERTICAL_PREFIX = {
    VerticalPosition.TOP: "upper",
    VerticalPosition.MIDDLE: "front",
    VerticalPosition.BOTTOM: "lower",
}


class PositionAnalyzer:
    """Maps a bounding box to an approximate spatial position within the image.

    Based solely on where the bbox's center falls within a 3x3 grid of the
    image — no physical distance, GPS, or depth is inferred.
    """

    def analyze(self, bbox: BoundingBox, image_width: int, image_height: int) -> Position:
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2

        horizontal = self._horizontal(center_x, image_width)
        vertical = self._vertical(center_y, image_height)
        region = Region(f"{_REGION_VERTICAL_PREFIX[vertical]}-{horizontal.value}")

        return Position(horizontal=horizontal, vertical=vertical, region=region)

    @staticmethod
    def _horizontal(center_x: float, image_width: int) -> HorizontalPosition:
        third = image_width / 3
        if center_x < third:
            return HorizontalPosition.LEFT
        if center_x < 2 * third:
            return HorizontalPosition.CENTER
        return HorizontalPosition.RIGHT

    @staticmethod
    def _vertical(center_y: float, image_height: int) -> VerticalPosition:
        third = image_height / 3
        if center_y < third:
            return VerticalPosition.TOP
        if center_y < 2 * third:
            return VerticalPosition.MIDDLE
        return VerticalPosition.BOTTOM
