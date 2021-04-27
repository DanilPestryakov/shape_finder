import argparse
import cv2
import numpy as np
import warnings


class ShapeFinder:
    def __init__(self):
        self.WIDTH_BORDER = 50
        self.APPROX_EPS = 2e-2
        self.TOL = 1e-6
        self.detect_shapes = []
        self.primitives_list = []

    def read_primitives(self, file_path):
        with open(file_path) as file:
            n = int(file.readline())
            for i in range(n):
                primitive_coords = list(map(int, file.readline().split(', ')))
                primitive_vertexes = np.array(primitive_coords).reshape((-1, 2))
                if cv2.contourArea(primitive_vertexes, True) < 0:
                    primitive_vertexes = primitive_vertexes[::-1]
                self.primitives_list.append(primitive_vertexes)

    def _clean_noise_from_image(self, image):
        image = cv2.copyMakeBorder(image, self.WIDTH_BORDER, self.WIDTH_BORDER, self.WIDTH_BORDER, self.WIDTH_BORDER,
                                   cv2.BORDER_CONSTANT, value=0)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)[hierarchy[0, :, 3] != -1]
        new_image = np.ones_like(image)
        cv2.drawContours(new_image, contours, -1, 0, -1)
        image[new_image.astype(bool)] = 255
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def detect_shape_on_image(self, image, with_clean_noise=True):
        if with_clean_noise:
            image = self._clean_noise_from_image(image)
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:-1]:
            shape = cv2.approxPolyDP(contour, self.APPROX_EPS * cv2.arcLength(contour, True), True)
            self.detect_shapes.append(np.squeeze(shape, 1))
        self.detect_shapes = np.array(self.detect_shapes)
        return self._find_transform_list()

    def _find_transform_list(self):
        transforms = []
        for detect_shape in self.detect_shapes:
            min_max_dist = np.inf
            for _ in range(self.detect_shapes.shape[0]):
                for j, primitive in enumerate(self.primitives_list):
                    if detect_shape.shape[0] == primitive.shape[0]:
                        scale = np.sqrt(cv2.contourArea(detect_shape) / cv2.contourArea(primitive))
                        angle = self._find_angle(detect_shape, primitive)
                        bias_x, bias_y = self._find_bias(detect_shape, primitive, scale, angle)
                        max_dist = self._find_max_dist(detect_shape, primitive, scale, angle, bias_x, bias_y)
                        if max_dist < min_max_dist:
                            min_max_dist = max_dist
                            min_j, min_bias_x, min_bias_y, min_scale, min_angle = j, bias_x, bias_y, scale, angle
                detect_shape = np.roll(detect_shape, 1, 0)
            if min_max_dist < np.inf:
                transforms.append([min_j, min_bias_x - self.WIDTH_BORDER, min_bias_y - self.WIDTH_BORDER,
                                   min_scale, min_angle * 180 / np.pi])
        return np.round(transforms).astype(int)

    def _find_angle(self, detect_shape, primitive):
        x_p, y_p = (detect_shape[1] - detect_shape[0]) / np.linalg.norm(detect_shape[1] - detect_shape[0])
        x_s, y_s = (primitive[1] - primitive[0]) / np.linalg.norm(primitive[1] - primitive[0])
        if np.abs(x_s) <= self.TOL:
            sin_angle, cos_angle = -x_p / y_s, y_p / y_s
        elif np.abs(y_s) <= self.TOL:
            sin_angle, cos_angle = y_p / x_s, x_p / x_s
        else:
            sin_angle, cos_angle = (y_p / y_s - x_p / x_s) / (x_s / y_s + y_s / x_s), \
                                (x_p / y_s + y_p / x_s) / (x_s / y_s + y_s / x_s)
        angle = np.arcsin(sin_angle)
        if cos_angle < 0:
            angle = np.pi - angle
        return angle

    def _find_bias(self, detect_shape, primitive, scale, angle):
        x_p, y_p = detect_shape[0]
        x_s, y_s = primitive[0] * scale
        bias_x, bias_y = x_p - (x_s * np.cos(angle) - y_s * np.sin(angle)), \
                        y_p - (x_s * np.sin(angle) + y_s * np.cos(angle))
        return bias_x, bias_y

    def _find_max_dist(self, detect_shape, primitive, scale, angle, bias_x, bias_y):
        max_dist = 0
        for figure_vertex, template_vertex in zip(detect_shape, primitive):
            x_s, y_s = template_vertex * scale
            x_s_t, y_s_t = bias_x + (x_s * np.cos(angle) - y_s * np.sin(angle)), \
                        bias_y + (x_s * np.sin(angle) + y_s * np.cos(angle))
            max_dist = max(max_dist, np.linalg.norm(figure_vertex - (x_s_t, y_s_t)))
        return max_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find shapes in the image')
    parser.add_argument('-s', '--shapes_file_name', help='Shapes file name')
    parser.add_argument('-i', '--image_name', help='Image name')
    args = parser.parse_args()

    image = cv2.imread(args.image_name, cv2.IMREAD_GRAYSCALE)

    finder = ShapeFinder()

    finder.read_primitives(args.shapes_file_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transforms = finder.detect_shape_on_image(image)

    print(transforms.shape[0])
    for transform in transforms:
        print(*transform, sep=', ')
