import os
import os.path

def get_testdata(*filename):
    import pysm
    # if a test data dir is set by run-tests.py, use it.
    # otherwise fall back to the test data in source (e.g. if ran directly with py.test)
    reldir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(pysm.__file__), '..')), "test_data")
    testdata = os.environ.get('PYSM_TESTDATA_DIR', reldir)
    return os.path.join(testdata, *filename)
