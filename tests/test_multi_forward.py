from tempfile import TemporaryDirectory
import unittest

from models.multi_forward_tacotron import MultiForwardTacotron


class TestMultiForwardTacotron(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(prefix='TestForwardDatasetTmp')

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_items(self) -> None:
        multi_forward_taco = MultiForwardTacotron.from_config(config)
