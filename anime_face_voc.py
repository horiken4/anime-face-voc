# conding: utf-8

import os
import argparse

import cv2
import random


class AnimeFaceDetector(object):
    def __init__(self, cascade_file):
        self._cascade = cv2.CascadeClassifier(cascade_file)

    def detect(self, img):
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24), maxSize=(640, 640))

        # Display detected faces
        """
        face_img = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.namedWindow('faces')
        cv2.imshow('faces', face_img)
        cv2.waitKey(0)
        """

        if len(faces) == 0:
            return None

        return Annotation(faces, ['animeface'] * len(faces))


class Annotation(object):
    def __init__(self, rects, classes):
        self._folder = ''
        self._filename = ''
        self._width = 0
        self._height = 0
        self._depth = 0
        self._rects = rects
        self._classes = classes

    def serialize(self):
        objects = []
        for i, r in enumerate(self._rects):
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            objects.append({
                'bndbox': {
                    'xmin': str(x),
                    'ymin': str(y),
                    'xmax': str(x + w),
                    'ymax': str(y + h)
                },
                'name': self._classes[i],
                'pose': 'Unspecified',
                'truncated': '0',
                'difficult': '0'
            })

        value = {
            'annotation': {
                'folder': self._folder,
                'filename': self._filename,
                'source': {
                    'database': 'Anime Face Database',
                    'annotation': 'Anime Face',
                    'image': 'flicker',
                    'flickerid': '0'
                },
                'owner': {
                    'flickerid': '0',
                    'name': 'unknown'
                },
                'size': {
                    'width': self._width,
                    'height': self._height,
                    'depth': self._depth
                },
                'segmented': '0',
                'object': objects
            }
        }

        return '%YAML:1.0\n' + self._dump(None, value, 0)

    def _dump(self, parent, value, depth):
        if isinstance(value, int) or isinstance(value, long) or isinstance(value, float):
            return str(value) + '\n'
        elif isinstance(value, str) or isinstance(value, unicode):
            return '\'' + value + '\'\n'
        elif isinstance(value, dict):
            if isinstance(parent, list):
                s = ''
                indent = ''
                for k in value:
                    v = value[k]
                    s += indent + k + ': ' + self._dump(value, v, depth + 1)
                    indent = '  ' * depth
                return s
            else:
                s = '\n'
                indent = '  ' * depth
                for k in value:
                    v = value[k]
                    s += indent + k + ': ' + self._dump(value, v, depth + 1)
                return s
        elif isinstance(value, list):
            s = '\n'
            for i, v in enumerate(value):
                s += '  ' * depth + '- ' + self._dump(value, v, depth + 1)
            return s
        else:
            raise Exception('Not implemented ({0})'.format(type(value)))

    @property
    def classes(self):
        return self._classes

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, v):
        self._folder = v

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        self._filename = v

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        self._width = v

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, v):
        self._height = v

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, v):
        self._depth = v


class VocDataset(object):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._annotation_dir = root_dir + '/Annotations'
        self._jpegimages_dir = root_dir + '/JPEGImages'
        self._imagesets_dir = root_dir + '/ImageSets/Main'

        self._classes = []
        self._trains = []
        self._tests = []

        if not os.path.isdir(self._root_dir):
            os.mkdir(root_dir)
        if not os.path.isdir(self._annotation_dir):
            os.mkdir(self._annotation_dir)
        if not os.path.isdir(self._jpegimages_dir):
            os.mkdir(self._jpegimages_dir)
        if not os.path.isdir(self._imagesets_dir):
            os.makedirs(self._imagesets_dir)

    def save_annotation(self, annotation, img):
        is_train = True
        if random.randint(1, 100) > 80:
            is_train = False

        _, annotation.folder = os.path.split(self._root_dir)

        # Save image
        cv2.imwrite(self._jpegimages_dir + '/' + annotation.filename, img)

        # Save annotation
        name, _ = os.path.splitext(os.path.basename(annotation.filename))
        with open(self._annotation_dir + '/' + name + '.yml', 'w') as f:
            f.write(annotation.serialize())

        # Listup classes

        for cls in annotation.classes:
            if cls not in self._classes:
                self._classes.append(cls)

        if is_train:
            self._trains.append(name + '\n')
        else:
            self._tests.append(name + '\n')

    def save(self):
        with open(self._imagesets_dir + '/train.txt', 'w') as f:
            f.writelines(self._trains)
        with open(self._imagesets_dir + '/test.txt', 'w') as f:
            f.writelines(self._tests)
        with open(self._imagesets_dir + '/class.txt', 'w') as f:
            f.writelines(self._classes)


class VocDatasetMaker(object):
    def __init__(self):
        pass

    def make(self, list_file, odir):
        dataset = VocDataset(odir)
        detector = AnimeFaceDetector('./lbpcascade_animeface.xml')

        image_paths = self._load_image_list(list_file)

        for path in image_paths:
            print path

            name, _ = os.path.splitext(os.path.basename(path))

            img = cv2.imread(path, 1)
            annotation = detector.detect(img)
            if annotation is None:
                continue
            annotation.filename = name + '.jpg'
            annotation.width = img.shape[1]
            annotation.height = img.shape[0]
            annotation.depth = img.shape[2]
            dataset.save_annotation(annotation, img)

        dataset.save()

    def _load_image_list(self, list_file):
        paths = []
        with open(list_file, 'r') as f:
            paths = f.readlines()
        for i, _ in enumerate(paths):
            paths[i] = paths[i].strip()
        return paths


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=
"""
usage: python anime_face_voc.py --list=$HOME/anime/images.txt --odir=./anime_face_voc
""")
    parser.add_argument('--listfile', metavar='listfile', required=True, help='Image list file')
    parser.add_argument('--odir', metavar='odir', required=True, help='Output directory')

    args = parser.parse_args()

    maker = VocDatasetMaker()
    maker.make(args.listfile, args.odir)

if __name__ == '__main__':
    main()
