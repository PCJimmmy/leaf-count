from PIL import Image
import os


'''
Rotate and mirror images to augment dataset by a factor of 8
'''


MULTIPLIER = 8


def main(input_path, output_path):
	filelist = sorted([f for f in os.listdir(input_path) if '.png' in f])
	num_images = len(filelist) * MULTIPLIER
	n = 0
	for f in filelist: 
		img = Image.open(input_path+'/'+f)
		for i in range(4): # 4x
			for mirror in [False, True]: # 2x
				n += 1
				img_copy = img.copy()
				if mirror:
					img_copy = img_copy.transpose(Image.FLIP_LEFT_RIGHT)
				img_copy.rotate(90*i).save('{}/{}.png'.format(output_path, n))
				if (n % 100 == 0) or (n == num_images):
					print("[{}/{}] Augmented the images and saved into '{}'.".format(n, num_images, output_path))
	print('')

		
