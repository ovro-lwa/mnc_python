import unittest

from mnc.settings import _effective_fft_shift


class TestEffectiveFftShift(unittest.TestCase):
    def test_unmodified_when_low_bits_match_firmware(self):
        self.assertEqual(_effective_fft_shift(2047), 2047)
        self.assertEqual(_effective_fft_shift(0x7FF), 2047)

    def test_firmware_masks_low_stages(self):
        # 0x1FFC default: lower 5 bits differ from hardcoded mask -> effective 0x1FFF
        self.assertEqual(_effective_fft_shift(0x1FFC), 0x1FFF)

    def test_masks_to_stage_width(self):
        self.assertEqual(_effective_fft_shift(0xFFFF), 0x1FFF)


if __name__ == '__main__':
    unittest.main()
