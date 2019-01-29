import os
import shutil
import glob


''' Remove the images from the selected directory ''' 


IDS_TO_REMOVE = ['11', '28', '61', '62', '90', '195', '219', '242', '274', '290', \
				 '337', '354', '370', '383', '388', '390', '415', '419']


def main(input_path, output_path):
	for f in glob.iglob(os.path.join(input_path, "*.png")):
		shutil.copy(f, output_path) 
	for f in os.listdir(output_path): 
		if f.replace('.png', '') in IDS_TO_REMOVE:
			os.remove(output_path+'/'+f) 