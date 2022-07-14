import os
import site
import sys
if not sys.prefix:
    # For cx_Freeze - this needs to be set for kivy to get sdl,glew,angle DLLs
    abspath = os.path.abspath(".")
    sys.prefix = abspath
    site.USER_BASE = abspath

from src import gui

if __name__ == '__main__':
    app = gui.LooperApp()
    try:
        app.run()
    finally:
        app.cleanup()
