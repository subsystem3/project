import os

from jupyter_core.paths import jupyter_config_dir
from jupyter_server.services.contents.filemanager import FileContentsManager
from traitlets.config import Config

# define the configuration directory
config_dir = jupyter_config_dir()

# define the configuration file path
config_file = os.path.join(config_dir, "jupyter_notebook_config.py")

# create a new Config instance
c = Config()

# set the JupyterLabCodeFormatter options
c.NotebookApp.contents_manager_class = FileContentsManager
c.ContentsManager.default_jupytext_formats = "ipynb,py"
c.NotebookApp.disable_check_xsrf = True
c.JupyterLabCodeFormatter.format_on_save = True
c.JupyterLabCodeFormatter.factory = "autopep8"
c.LatexExporter.template_file = "classic"
