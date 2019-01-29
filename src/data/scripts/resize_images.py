import argparse
import os
from PIL import Image


'''
Resize images to square dimensions (default 256x256)
'''


def resize_image(image, target_image_dimensions):
    """Resize an image to the given size."""
    return image.resize(target_image_dimensions, Image.ANTIALIAS)


def main(input_path, output_path, target_image_dimensions):
    """Resize the images in 'input_path' and save into 'output_path'."""

    images = [f for f in os.listdir(input_path) if '.png' in f]
    num_images = len(images)
    for i, image in enumerate(images):
        if '.png' in image:
            with open(os.path.join(input_path, image), 'r+b') as f:
                with Image.open(f) as img:
                    img = resize_image(img, target_image_dimensions)
                    img.save(os.path.join(output_path, image), img.format)
            if ((i+1) % 100 == 0) or (i+1 == num_images):
                print("[{}/{}] Resized the images and saved into '{}'.".format(i+1, num_images, output_path))
    print('')



    