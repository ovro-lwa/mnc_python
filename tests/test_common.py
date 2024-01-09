import unittest
from unittest.mock import patch

from mnc.common import ExceptionalSnap2FengineEtcdControl

class TestExceptionalSnap2FengineEtcdControl(unittest.TestCase):
    @patch('lwa_f.snap2_feng_etcd_client.Snap2FengineEtcdControl._send_command_etcd', autospec=True)
    def test_err_one_fid(self, mock_parent_send_command_etcd):
        mock_parent_send_command_etcd.return_value = {}

        ec = ExceptionalSnap2FengineEtcdControl(retries=1)

        with self.assertRaises(RuntimeError):
            ec.send_command(1, 'feng', 'is_programmed')

        mock_parent_send_command_etcd.assert_called_with(ec, 1, 'feng', 'is_programmed', {}, 10.0, 1)
        self.assertEqual(mock_parent_send_command_etcd.call_count, 2)

    @patch('lwa_f.snap2_feng_etcd_client.Snap2FengineEtcdControl._send_command_etcd', autospec=True)
    def test_no_retry(self, mock_parent_send_command_etcd):
        ec = ExceptionalSnap2FengineEtcdControl(retries=1)
        mock_parent_send_command_etcd.return_value = {}

        with self.assertRaises(RuntimeError):
            ec.send_command(1, 'feng', 'initialize', kwargs={'read_only': False})
        self.assertEqual(mock_parent_send_command_etcd.call_count, 1)

    @patch('lwa_f.snap2_feng_etcd_client.Snap2FengineEtcdControl._send_command_etcd', autospec=True)
    def test_err_all_fid(self, mock_parent_send_command_etcd):
        ec = ExceptionalSnap2FengineEtcdControl(retries=1)
        mock_parent_send_command_etcd.return_value = {}

        with self.assertRaises(RuntimeError):
            ec.send_command(0, 'feng', 'is_programmed')

        mock_parent_send_command_etcd.assert_called_with(ec, 0, 'feng', 'is_programmed', {}, 10.0, 11)
        self.assertEqual(mock_parent_send_command_etcd.call_count, 2)

    @patch('lwa_f.snap2_feng_etcd_client.Snap2FengineEtcdControl._send_command_etcd', autospec=True)
    def test_one_fid(self, mock_parent_send_command_etcd):
        mock_parent_send_command_etcd.return_value = None

        ec = ExceptionalSnap2FengineEtcdControl(retries=1)

        ec.send_command(1, 'feng', 'is_programmed')

        mock_parent_send_command_etcd.assert_called_with(ec, 1, 'feng', 'is_programmed', {}, 10.0, 1)
        self.assertEqual(mock_parent_send_command_etcd.call_count, 1)

    @patch('lwa_f.snap2_feng_etcd_client.Snap2FengineEtcdControl._send_command_etcd', autospec=True)
    def test_all_fid(self, mock_parent_send_command_etcd):
        ans =dict((k, None) for k in range(1,12))
        mock_parent_send_command_etcd.return_value = ans

        ec = ExceptionalSnap2FengineEtcdControl(retries=1)

        resp = ec.send_command(0, 'feng', 'is_programmed')

        mock_parent_send_command_etcd.assert_called_with(ec, 0, 'feng', 'is_programmed', {}, 10.0, 11)
        self.assertEqual(mock_parent_send_command_etcd.call_count, 1)
        self.assertEqual(resp, ans)