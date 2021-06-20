import xml.etree.ElementTree as ET

def load_voc_xml(annotation_path):
	root = ET.parse(annotation_path).getroot()
	size_node = root.find('size')

	width = int(float(size_node.find('width').text))
	height = int(float(size_node.find('height').text))
	depth = int(float(size_node.find('depth').text))

	annotation = {'size':{'height':height, 'width':width, 'depth':depth}, 'boxes':[]}
	for obj in root.findall('object'):
		temp_dict = {}

		bndbox = obj.find('bndbox')
		temp_dict['name'] = obj.find('name').text
		temp_dict['x_min'] = int(float(bndbox.find('xmin').text))
		temp_dict['y_min'] = int(float(bndbox.find('ymin').text))
		temp_dict['x_max'] = int(float(bndbox.find('xmax').text))
		temp_dict['y_max'] = int(float(bndbox.find('ymax').text))

		annotation['boxes'].append(temp_dict)
	return annotation