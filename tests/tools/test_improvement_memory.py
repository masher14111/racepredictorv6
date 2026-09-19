"""Offline checks for extending the completed programme without accepting broken ordering."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/improvement_memory.py"
SPEC = importlib.util.spec_from_file_location("improvement_memory", SCRIPT)
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)


class ManifestExtensionTests(unittest.TestCase):
    def manifest(self, count):
        return {"stage_count": count, "steps": [{"id": f"{i:02d}"} for i in range(1, count + 1)]}

    def test_original_and_extended_manifest_are_valid(self):
        for count in (18, 24):
            self.assertEqual(memory.manifest_order_errors(self.manifest(count)), [])

    def test_gaps_duplicates_and_count_mismatch_fail(self):
        for index, replacement in ((18, "20"), (19, "19")):
            data = self.manifest(24)
            data["steps"][index]["id"] = replacement
            self.assertTrue(memory.manifest_order_errors(data))
        data = self.manifest(24)
        data["stage_count"] = 23
        self.assertTrue(memory.manifest_order_errors(data))

    def test_invalid_count_and_removal_of_original_programme_fail(self):
        for count in (17, 100, 24.0, "24", True, None):
            data = self.manifest(24)
            data["stage_count"] = count
            self.assertTrue(memory.manifest_order_errors(data))


if __name__ == "__main__":
    unittest.main()
