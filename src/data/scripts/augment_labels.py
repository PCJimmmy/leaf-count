import os
import csv


'''
Copy each label 8 times, corresponding to image dataset augmentation
'''


MULTIPLIER = 8


def main(input_path, output_path):
	counts_list = []
	with open(input_path+'/'+'labels.csv', 'r') as f: 
		reader = csv.DictReader(f, fieldnames=['image_url', 'minimum', 'maximum'])
		next(reader)  # Skip header
		for row in reader:
			image_id = row['image_url'].replace('https://s3.amazonaws.com/plant-science-lab/mechanical+turk/plant_', '').replace('.png', '')
			counts_list.append((image_id, row['minimum'], row['maximum']))
		counts_list = sorted(counts_list, key=lambda x: x[0])

	n = 0
	with open(output_path+'/'+'labels.csv', 'w') as g: 
		writer = csv.writer(g)
		writer.writerow(['image_id', 'minimum', 'maximum'])
		for ID, minimum, maximum in counts_list: 
			for i in range(MULTIPLIER):
				n += 1
				writer.writerow([n, minimum, maximum])
