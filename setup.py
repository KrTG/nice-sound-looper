"""Package everything for windows"""

import os
import shutil
import subprocess

try:
    shutil.rmtree("build")
except OSError as e:
    pass

try:
    os.mkdir("build")
except OSError:
    pass
result = subprocess.run(
    ["poetry", "env", "list", "--full-path"], stdout=subprocess.PIPE, shell=True
)

env_path = result.stdout.decode().split()[0]
shutil.copytree(env_path, os.path.join("build", "env"))
shutil.copytree("src", os.path.join("build", "src"))
shutil.copy2("main.py", os.path.join("build", "main.py"))
os.mkdir(os.path.join("build", "autosaves"))
with open(os.path.join("build", "run.bat"), "w") as runfile:
    runfile.write("env\Scripts\python.exe main.py\n")
