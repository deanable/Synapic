import unittest
from unittest.mock import MagicMock

from src.core.daminion_client import DaminionClient


class TestDaminionClientVersionControl(unittest.TestCase):
    def test_checkout_item_calls_version_control(self):
        client = DaminionClient("https://example.net", "user", "pass")
        client._api.version_control = MagicMock()
        client._api.version_control.checkout.return_value = True

        result = client.checkout_item(42)

        self.assertTrue(result)
        client._api.version_control.checkout.assert_called_once_with([42])

    def test_checkin_item_calls_version_control(self):
        client = DaminionClient("https://example.net", "user", "pass")
        client._api.version_control = MagicMock()

        result = client.checkin_item(42, "C:\\temp\\upscaled.jpg")

        self.assertTrue(result)
        client._api.version_control.checkin.assert_called_once_with(
            42, "C:\\temp\\upscaled.jpg"
        )

    def test_checkout_item_handles_errors(self):
        client = DaminionClient("https://example.net", "user", "pass")
        client._api.version_control = MagicMock()
        client._api.version_control.checkout.side_effect = RuntimeError("fail")

        result = client.checkout_item(42)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
