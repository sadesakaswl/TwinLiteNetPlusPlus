import unittest
from unittest.mock import patch

import numpy as np

from BDD100K import random_crop


class RandomCropTest(unittest.TestCase):
    def test_applies_one_crop_to_image_and_masks(self):
        mask = np.arange(30).reshape(5, 6)
        image = np.stack((mask, mask, mask), axis=-1)
        with patch("BDD100K.random.randint", side_effect=[2, 1]):
            cropped_image, mask1, mask2 = random_crop((2, 3), image, mask, mask + 100)

        expected = mask[2:4, 1:4]
        np.testing.assert_array_equal(cropped_image, image[2:4, 1:4])
        np.testing.assert_array_equal(mask1, expected)
        np.testing.assert_array_equal(mask2, expected + 100)

    def test_rejects_oversized_crop_and_misaligned_mask(self):
        image = np.zeros((5, 6, 3))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            random_crop((6, 3), image, np.zeros((5, 6)))
        with self.assertRaisesRegex(ValueError, "matching spatial dimensions"):
            random_crop((2, 3), image, np.zeros((4, 6)))
