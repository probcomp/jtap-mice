import unittest
import os

if __name__ == '__main__':
    # Define the directory containing tests. Assuming 'tests' is right next to this script.
    tests_dir = os.path.join(os.path.dirname(__file__), 'unit_tests')
    
    # Create a test loader
    loader = unittest.TestLoader()
    # Discover and load all tests in the specified directory and its subdirectories
    suite = loader.discover(start_dir=tests_dir, pattern='test_*.py')
    
    # Create a test runner that will run the discovered tests
    runner = unittest.TextTestRunner()
    # Run the tests
    runner.run(suite)


