import unittest
from unittest.mock import patch
from mnc.control import Controller

class TestController(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()

    def test_init_with_args(self):
        controller = Controller(recorders='dr5')
        self.assertEqual(controller.conf['dr']['recorders'], ['dr5'])

if __name__ == '__main__':
    unittest.main()
    
