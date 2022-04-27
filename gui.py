
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics import Color
from kivy.graphics import Rectangle


class Screen(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start_listening(self):
        pass

class LooperApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    LooperApp().run()
