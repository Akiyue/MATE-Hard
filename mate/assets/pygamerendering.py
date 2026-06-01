"""
2D rendering framework
"""

# pylint: skip-file

import os
import sys

import PIL.Image
import numpy as np
from skimage.draw import polygon_perimeter, polygon, circle_perimeter, disk, line, line_aa

if 'Apple' in sys.version and 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'


import pygame

import PIL

RAD2DEG = 57.29577951308232

class Geom:
    def __init__(self):
        self.color = pygame.Color(0, 0, 0, 255)
        self.transform = Transform()
        self.attrs: dict = {}


    def reset(self):
        self.transform = Transform()
        self.attrs = {}

    def render(self, surf: pygame.Surface, numpy_array = False):
        raise NotImplementedError

    def add_attr(self, key, value):
        self.attrs[key] = value

    def set_color(self, r, g, b, a=255):
        r = self.toRGB(r)
        g = self.toRGB(g)
        b = self.toRGB(b)
        a = self.toRGB(a)
        if int(r) != r:
            r = int(r*255)
        if int(g) != g:
            g = int(g*255)
        if int(b) != b:
            b = int(b*255)
        if int(a) != a:
            a = int(a*255)

        self.color.update(r, g, b, a)

    def set_alpha(self, a=1):
        a = self.toRGB(a)
        self.color.a = a

    @staticmethod
    def toRGB(value):
        if value <= 1 and type(value) is float:
            return int(value*255)
        return value

    def move(self, V):
        return V + self.transform.translation
        # return [[x + self.transform.translation[0], y + self.transform.translation[1]] for x, y in V]

    def rotate(self, V):
        # print(V.shape)
        cos = np.cos(-self.transform.rotation)
        sin = np.sin(-self.transform.rotation)
        return np.dot(np.array([[cos, sin],
                                [-sin, cos]]),
                    V.T).T

        # return [[x*cos + y*sin, -x*sin + y*cos] for x, y in V]

    def scale(self, V):
        return V * self.transform.scale
        # return [[x * self.transform.scale[0], y * self.transform.scale[1]] for x, y in V]

    def add_transform(self, transform):
        self.transform.add(transform)

    def sub_transform(self, transform):
        self.transform.sub(transform)

    def draw2numpy(self, surf, rr, cc):
        alpha = self.color.a / 255
        color = np.array([self.color.r, self.color.g, self.color.b], dtype=np.uint8)
        surf[rr, cc] = surf[rr, cc] * (1-alpha) + color*alpha


class Transform:
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale = (1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        pass

    def disable(self):
        pass

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


    def add(self, transform):
        self.set_translation(self.translation[0] + transform.translation[0], self.translation[1] + transform.translation[1])
        self.set_rotation(self.rotation + transform.rotation)
        self.set_scale(self.scale[0] * transform.scale[0], self.scale[1] * transform.scale[1])

    def sub(self, transform):
        self.set_translation(self.translation[0] - transform.translation[0], self.translation[1] - transform.translation[1])
        self.set_rotation(self.rotation - transform.rotation)
        self.set_scale(self.scale[0] / transform.scale[0], self.scale[1] / transform.scale[1])

class Polygon(Geom):
    def __init__(self, v: np.ndarray, width = False):
        Geom.__init__(self)
        self.v = v
        self.width = width

    def draw_polygon_alpha(self, surface: pygame.Surface, points):
        lx, ly = zip(*points)
        min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
        target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(shape_surf, self.color, [(x - min_x, y - min_y) for x, y in points], self.width)
        surface.blit(shape_surf, target_rect)

    def render(self, surf: pygame.Surface | np.ndarray, numpy_array = False):
        V = self.rotate(self.v)
        V = self.scale(V)
        V = self.move(V)

        if numpy_array:
            if not self.width:
                rr, cc = polygon(V[:, 0], V[:,1], surf.shape)
            else:
                rr, cc = polygon_perimeter(V[:, 0], V[:,1], surf.shape, True)

            self.draw2numpy(surf, rr, cc)

            return

        if self.color[3] != 255:
            self.draw_polygon_alpha(surf, V)

        else:
            pygame.draw.polygon(surf, self.color, V, self.width)

    def set_linewidth(self, x):
        self.width = x

class Circle(Geom):
    def __init__(self, v: np.ndarray, radius, width = False):
        Geom.__init__(self)
        self.v = v
        self.r = radius
        self.width = width

    def draw_circle_alpha(self, surface: pygame.Surface, center, radius):
        target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, self.color, (radius, radius), radius, self.width)
        surface.blit(shape_surf, target_rect)

    def scale(self, R):
        return R*self.transform.scale[0]

    def move(self, V):
        return V + self.transform.translation

    def render(self, surf: pygame.Surface | np.ndarray, numpy_array = False):
        R = self.scale(self.r)
        V = self.move(self.v)

        if numpy_array:
            if not self.width:
                rr, cc = disk(V, R, shape=surf.shape)
            else:
                rr, cc = circle_perimeter(V[0], V[1], R, shape=surf.shape)

            self.draw2numpy(surf, rr, cc)
            return

        if self.color[3] != 255:
            self.draw_circle_alpha(surf, V, R)
        else:
            pygame.draw.circle(surf, self.color, V, R, self.width)

    def set_linewidth(self, x):
        self.width = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.width = 1

    def render(self, surf: pygame.Surface, numpy_array = False):
        if not numpy_array:
            pygame.draw.line(surf, self.color, self.start, self.end, int(self.width))
        else:
            if not self.width:
                rr, cc = line(self.start[0], self.start[1], self.end[0], self.end[1])
            else:
                rr, cc = line_aa(self.start[0], self.start[1], self.end[0], self.end[1])
            self.draw2numpy(surf, rr, cc)
            # Handle numpy array rendering
            return

class Compound(Geom):
    def __init__(self, gs: list[Geom]):
        Geom.__init__(self)
        self.gs = gs

    def render(self, surf: pygame.Surface, numpy_array = False):
        for g in self.gs:
            g.add_transform(self.transform)
            g.render(surf, numpy_array)
            g.sub_transform(self.transform)

class PygameImage(pygame.sprite.Sprite):
    def __init__(self, PIL_IMAGE: PIL.Image.Image, center = (0, 0)):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.fromstring(PIL_IMAGE.tobytes(), PIL_IMAGE.size, "RGBA")

        # self.original_image = pygame.image.load(image_file).convert_alpha()
        width, height = self.original_image.get_size()
        self.image = self.original_image
        self.rect = self.image.get_rect(topleft=[center[0] - width//2, center[1] - height//2])

    def transform(self, transform: Transform):
        if transform.scale[0] != 1 or transform.scale[1] != 1:
            self.image = pygame.transform.scale_by(self.original_image, transform.scale)
        else:
            self.image = self.original_image

        self.rect.center = transform.translation

        if transform.rotation != 0:
            self.image = pygame.transform.rotate(self.image, -transform.rotation * RAD2DEG)

        self.rect = self.image.get_rect(center = transform.translation)

    def rotate(self, angle):
        self.image = pygame.transform.rotate(self.original_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def move(self, location):
        self.rect.center = location

    def scale(self, scale):
        if scale[0] != 1 or scale[1] != 1:
            self.image = pygame.transform.scale_by(self.original_image, scale)
            self.rect = self.image.get_rect(center=self.rect.center)


class Image(Geom):
    def __init__(self, fname, width = None, height = None):
        Geom.__init__(self)
        self.set_color(1.0, 1.0, 1.0)
        self.width = width
        self.height = height
        self.fname = fname
        # self.img
        img: PIL.Image.Image = PIL.Image.open(self.fname).convert("RGBA")
        # print(img.format)
        if width is None:
            self.width = img.size[0]
        if height is None:
            self.height = img.size[1]


        self.width = int(self.width)
        self.height = int(self.height)

        # if width and height:

        self.img = img.resize((self.width, self.height))

        # self.img = pygame.transform.scale(img, (width, height))
        self.flip = False
        self.v = np.array([[-self.width//2, -self.height//2], [self.width//2, self.height//2]])
        self.initialize = False

    def render(self, surf: pygame.Surface, numpy_array = False):
        if not numpy_array:
            if not self.initialize:
                self.img = PygameImage(self.img)
                self.initialize = True

            self.img.transform(self.transform)

            surf.blit(self.img.image, self.img.rect)
        else:
            if not self.initialize:
                self.img = np.rot90(self.img)
                self.initialize = True

            color = self.img[:, :, :3].astype(np.uint8)
            alpha = np.expand_dims(self.img[:, :, 3], -1)/255
            V = self.move(self.v).astype(np.uint32)
            surf[V[0][0]: V[1][0], V[0][1]: V[1][1]] = surf[V[0][0]: V[1][0], V[0][1]: V[1][1]] * (1-alpha) + color * alpha

def make_circle(radius=10, res=30, filled=True):
    return Circle(np.array([0, 0]), radius, not filled)



def make_polygon(v, filled=True):
    if filled:
        return Polygon(v)
    else:
        return make_polyline(v)


def make_polyline(v):
    return Polygon(v, True)


def make_capsule(length, width):
    left, right, top, bottom = 0, length, width / 2, -width / 2
    box = make_polygon([(left, bottom), (left, top), (right, top), (right, bottom)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_transform(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Viewer:
    def __init__(self, width, height, mode = 'human'):
        self.isopen = True

        self.width = width
        self.height = height

        if mode == 'human':
            self.screen = pygame.display.set_mode((width, height))

        self.rgb_array = (mode == 'rgb_array' or mode == 'numpy_array')
        self.numpy_array = (mode == 'numpy_array')

        if not self.numpy_array:
            pygame.init()

            self.fixed_window = pygame.Surface((width, height), pygame.SRCALPHA)
            self.fixed_window.fill((255, 255, 255, 255))
            pygame.event.set_allowed([pygame.QUIT])
        else:
            self.fixed_window = np.full((width, height, 3), 255, np.uint8)




        self.clock = pygame.time.Clock()
        self.geoms: list[Geom] = []
        self.onetime_geoms: list[Geom] = []
        self.transform = Transform()

        self.size = (width, height)
        self.scale = (1, 1)

    def close(self):
        if self.isopen and sys.meta_path:
            self.isopen = False
            pygame.quit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        self.size = (int(right-left), int(top-bottom))

        if not self.numpy_array:
            self.fixed_window = pygame.Surface(self.size, pygame.SRCALPHA)
            self.fixed_window.fill((255, 255, 255, 255))
        else:
            self.fixed_window = np.full((*self.size, 3), 255, np.uint8)

        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)

        self.transform = Transform(
            translation=(-left , -bottom)
        )

        self.scale = scalex * (right-left), scaley*(top-bottom)

    def add_geom(self, geom: Geom):
        self.geoms.append(geom)
        geom.add_transform(self.transform)
        geom.render(self.fixed_window, self.numpy_array)
        geom.sub_transform(self.transform)

    def del_geom(self, geom: Geom):
        if geom not in self.geoms:
            raise ValueError("dont find geom")

        self.geoms.remove(geom)

        if not self.numpy_array:
            self.fixed_window = pygame.Surface(self.size, pygame.SRCALPHA)
            self.fixed_window.fill((255, 255, 255, 255))
        else:
            self.fixed_window = np.full((*self.size, 3), 255, np.uint8)

        for geom in self.geoms:
            geom.add_transform(self.transform)
            geom.render(self.fixed_window, self.numpy_array)
            geom.sub_transform(self.transform)

    def update_geom(self, geom: Geom):
        if geom not in self.geoms:
            raise ValueError("dont find geom")

        if not self.numpy_array:
            self.fixed_window = pygame.Surface(self.size, pygame.SRCALPHA)
            self.fixed_window.fill((255, 255, 255, 255))
        else:
            self.fixed_window = np.full((*self.size, 3), 255, np.uint8)

        for geom in self.geoms:
            geom.add_transform(self.transform)
            geom.render(self.fixed_window, self.numpy_array)
            geom.sub_transform(self.transform)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self) -> tuple[np.ndarray, bool]:
        self.clock.tick()
        print(self.clock.get_fps())

        window = self.fixed_window.copy()

        for geom in self.onetime_geoms:
            geom.add_transform(self.transform)
            geom.render(window, self.numpy_array)
            geom.sub_transform(self.transform)


        self.onetime_geoms.clear()


        if not self.rgb_array:
            window = pygame.transform.smoothscale(window, self.scale)

            self.screen.blit(window, (0, 0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.isopen = False
                    pygame.quit()

            return self.isopen

        else:
            if self.numpy_array:
                return window.transpose([1, 0, 2])
            else:
                return self.get_pixel_data(window, (50, 50), (2050, 2050))

    def get_pixel_data(self, surf: pygame.Surface, top_left, bottom_right):
        w = bottom_right[0] - top_left[0]
        h = bottom_right[1] - top_left[1]
        sub_surface = surf.subsurface(pygame.Rect(*top_left, w, h))
        arr = pygame.surfarray.pixels3d(sub_surface)
        return np.rot90(arr, -1, (0, 1))
        # pxarray = pygame.PixelArray(surf)
        # return pxarray

    def __del__(self):
        self.close()

def main():
    import matplotlib.pyplot as plt
    a = Viewer(800, 800, mode='rgb_array')
    bound = 1000 * 1.05

    a.set_bounds(-bound, bound, -bound, bound)


    margin = make_polygon(
        1000 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]), filled=True
    )

    margin.set_color(0, 0, 0)

    margin.set_linewidth(5)
    a.add_geom(margin)

    c = make_circle(100)
    c.set_color(255, 0, 0, 225)

    p = Image("test.jpg")

    p.transform.set_scale(1.8, 0.6)

    # a.add_geom(p)

    p = make_polygon([[-50, -50], [50, -50], [50, 50], [-50, 50]])

    p = make_polygon(
                        100
                        * np.array([(1.0, 0.0), (0.3, 0.6), (-0.8, 0.6), (-0.8, -0.6), (0.3, -0.6)])
                    )

    p.set_color(255, 255, 0, 255)


    x, y = (0, 0)

    # image = PygameImage("test.jpg", (200, 200))

    # a.add_geom(c)


    run = True
    while True:
        p.add_transform(Transform(rotation=np.pi/2))
        p.transform.set_translation(x, y)
        a.add_onetime(c)
        a.add_onetime(p)
        run = a.render()
        # run = a.render()

        plt.imshow(run)
        plt.show()

        # break


    # print(run)
        # x += 1
        # pygame.draw.polygon(a.screen, [255, 0, 0], [[10, 100], [100, 100], [100, 200]])

    # pygame.quit()


if __name__ == "__main__":
    main()

