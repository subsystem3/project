import os

from jupyter_core.paths import jupyter_config_dir
from jupyter_server.services.contents.filemanager import FileContentsManager
from traitlets.config import Config

# DEFINE THE CONFIGURATION DIRECTORY
config_dir = jupyter_config_dir()

# DEFINE THE CONFIGURATION FILE PATH
config_file = os.path.join(config_dir, "jupyter_notebook_config.py")

# INITIALIZE AND CONFIGURE THE CONFIGURATION
c = Config()
c.NotebookApp.contents_manager_class = FileContentsManager
c.ContentsManager.default_jupytext_formats = "ipynb,py"
c.NotebookApp.disable_check_xsrf = True
c.JupyterLabCodeFormatter.format_on_save = True
c.JupyterLabCodeFormatter.factory = "autopep8"
c.LatexExporter.template_file = "classic"
