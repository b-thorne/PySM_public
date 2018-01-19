from .pysm import Sky, Instrument
from .components import Dust, Synchrotron, AME, Freefree, CMB
from .common import read_map, convert_units

def get_template_dir():
    import os.path
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'template')
