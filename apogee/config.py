
import os


pwd = os.path.dirname(__file__)

def _absolute_data_dir(relative_path):
    data_dir = os.path.abspath(os.path.join(pwd, "../data/"))
    return os.path.abspath(os.path.join(data_dir, relative_path))

config = dict(
    DATA_DIR=_absolute_data_dir(""),
    APOGEE_DIR=_absolute_data_dir("dr14/apogee"),
    APOGEE_DR14_DIR=_absolute_data_dir("dr14/apogee/spectro/redux/r8/stars/"),
    CANNON_DR14_DIR=_absolute_data_dir("dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/cannon/")
    )
