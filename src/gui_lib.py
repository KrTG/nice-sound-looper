from kivy.properties import AliasProperty
from kivy.properties import ListProperty
from kivy.uix.image import Image

class StretchImage(Image):
    stretch_factor = ListProperty([1, 1])
    def get_norm_image_size(self):
        if not self.texture:
            return list(self.size)
        ratio = self.image_ratio
        w, h = self.size
        tw, th = self.texture.size
        stretch_w, stretch_h = self.stretch_factor

        # ensure that the width is always maximized to the container width
        return [max(tw, w * stretch_w), max(th, h * stretch_h)]

    norm_image_size = AliasProperty(get_norm_image_size,
                                bind=('texture', 'size', 'allow_stretch',
                                        'image_ratio', 'keep_ratio'),
                                cache=True)
