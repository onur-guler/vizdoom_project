from numpy.typing import NDArray
import numpy as np
import cv2


def img_write_text(
    img: NDArray,
    offset: tuple[int, int],
    text: str,
    scale: float = 1,
    thickness: int = 2,
) -> NDArray:
    """
    Draws text with the offset from top-left of img with white text
    """
    out = img.copy()
    if out.dtype != np.uint8:
        out = (out * 255).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # get text size (width, height) and baseline
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = offset[0]
    y = offset[1] + text_h  # y is baseline in putText

    # Draw main white text
    cv2.putText(
        out,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return out
