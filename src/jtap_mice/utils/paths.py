import os

def get_package_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def get_assets_dir():
    return os.path.join(get_package_dir(), "assets")

def get_fonts_dir():
    return os.path.join(get_assets_dir(), "fonts")

def get_tests_dir():
    return os.path.join(get_package_dir(), "tests")

def get_unit_tests_dir():
    return os.path.join(get_tests_dir(), "unit_tests")

# asserts during import
assert os.path.exists(get_package_dir())
assert os.path.exists(get_tests_dir())
assert os.path.exists(get_unit_tests_dir())
assert os.path.exists(get_fonts_dir())
