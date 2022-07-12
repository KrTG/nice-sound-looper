from src import gui

if __name__ == '__main__':
    app = gui.LooperApp()
    try:
        app.run()
    finally:
        app.cleanup()
