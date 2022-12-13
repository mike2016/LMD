
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)


# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
tools_path = osp.join(this_dir, '..', 'tools')
#util_path = osp.join(this_dir, '..', 'lib/util')
#add_path(util_path)
add_path(lib_path)
add_path(tools_path)


