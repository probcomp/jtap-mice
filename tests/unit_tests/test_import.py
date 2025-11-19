import unittest

class TestImportSNMCPhysics(unittest.TestCase):
    def test_import(self):
        try:
            import jtap_mice
            # If the import succeeds, assert True
            self.assertTrue(True)
        except ImportError:
            # If the import fails, assert False and provide a message
            self.assertTrue(False, "Failed to import jtap_mice package")

if __name__ == '__main__':
    unittest.main()
