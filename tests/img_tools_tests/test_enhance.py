import numpy as np

from img_tools.core.enhance import enhance_image


def test_enhancement_supports_threshold_and_brightness():
    image = np.array([[10, 100]], dtype=np.uint8)
    result = enhance_image(image, brightness=20, threshold=50)

    assert result.tolist() == [[0, 255]]
