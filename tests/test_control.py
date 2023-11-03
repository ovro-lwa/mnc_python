import unittest
from unittest.mock import patch
from mnc.control import Controller

class TestController(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()

    def test_init_no_args(self):
        with self.assertRaises(TypeError):
            Controller()

    def test_init_with_args(self):
        controller = Controller('arg1', 'arg2')
        self.assertEqual(controller.arg1, 'arg1')
        self.assertEqual(controller.arg2, 'arg2')

    @patch('control.Controller.dsa_store')
    def test_mock_dsa_store(self, mock_dsa_store):
        mock_dsa_store.return_value = 'mocked'
        result = self.controller.dsa_store('arg1', 'arg2')
        self.assertEqual(result, 'mocked')

if __name__ == '__main__':
    unittest.main()
    