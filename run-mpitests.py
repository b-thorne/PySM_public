# pop the current directory from search path
# python interpreter adds this to a top level script
# but we will likely have a name conflict (runtests.py .vs runtests package)
import sys; sys.path.pop(0)
from runtests.mpi import Tester
import os
import os.path

# export the path to the test data for tests to find.

srcdir = os.path.abspath(os.path.dirname(__file__))
os.environ['PYSM_TESTDATA_DIR'] = os.path.join(srcdir, 'test_data')

tester = Tester(os.path.join(os.path.abspath(__file__)), "pysm")

tester.main(sys.argv[1:])
