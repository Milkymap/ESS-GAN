import cv2 

import numpy as np 
import pickle as pk
import operator as op 

import torch as th 
import torchtext as tt 

from PIL import Image

from os import path   
from collections import Counter, OrderedDict 

from libraries.strategies import * 

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence 
from torchtext.vocab import vocab 
from torchvision import transforms as T 
from nltk.tokenize import RegexpTokenizer

class DATAHOLDER(Dataset):
	def __init__(self, path_to_storage, max_len=18, neutral='<###>', shape=(256, 256), nb_items=None, default_index=0):
		self.root = path_to_storage
		self.max_len = max_len
		self.neutral = neutral
		self.nb_items = nb_items
		self.default_index = default_index
		
		self.filenames, self.idmap, self.bounding_boxes = self.prepare()
		self.vocab_mapper, self.captions_mapper, self.num_embeddings = self.build_vocab()
		
		self.transform = T.Compose([
			T.Resize(shape),
			T.ToTensor(), 
			T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])

	def prepare(self):
		path_0 = path.join(self.root, 'images.txt')
		path_1 = path.join(self.root, 'bounding_boxes.txt')
		path_2 = path.join(self.root, 'filenames.pickle')

		line_0 = open(path_0, 'r').read().split('\n')
		line_1 = open(path_1, 'r').read().split('\n')
		line_2 = open(path_2, 'r').read().split('\n')
		line_3 = pk.load(open(path_2, 'rb'))

		idmap = dict([ elm.replace('.jpg', '').split(' ')[-1::-1] for elm in line_0 if len(elm) > 0 ])
		idmap = {key:int(val) - 1 for key,val in idmap.items()}
		bboxes = [ list(map(float, elm.split(' ')[1:])) for elm in line_1 if len(elm) > 0 ]
	
		filenames = [ f_name for f_name, f_id in idmap.items() if f_name in line_3 ]
		if self.nb_items is not None:
			filenames = filenames[:self.nb_items]
		return filenames, idmap, bboxes

	def crop_image(self, img, box):
		W, H = img.size
		x, y, w, h = box 
		r = np.maximum(w, h) * 0.75
		a = (2 * x + w) / 2
		b = (2 * y + h) / 2
		e = np.array([a - r, b - r, a + r, b + r])

		x_min, y_min = np.maximum(e[:2], [0, 0]).astype('int32')
		x_max, y_max = np.minimum(e[2:], [W, H]).astype('int32')
		return [x_min, y_min, x_max, y_max]

	def read_image(self, filename):
		path_2_img = path.join(self.root, 'images', f'{filename}.jpg')
		if path.isfile(path_2_img):
			image = Image.open(path_2_img).convert('RGB')
			box = list(map(int, self.bounding_boxes[int(self.idmap[filename])]))
			croped_box = self.crop_image(image, box)
			image = image.crop(croped_box)
			matrix = self.transform(image)
			return matrix 

	def build_vocab(self):
		counter = Counter()
		tokenizer = RegexpTokenizer(r'\w+')
		accumulator = []
		for f_name in self.filenames:
			fp = open(path.join(self.root, 'text', f'{f_name}.txt'), 'r')
			captions = fp.read().split('\n')  # list of captions
			tokenized_caps = []
			for cap in captions:
				if len(cap) > 0:
					tokens = tokenizer.tokenize(cap.lower())
					tokenized_caps.append(tokens)
					counter.update(tokens)

			accumulator.append( (f_name, tokenized_caps) )

		mapper = vocab(counter)
		mapper.insert_token(self.neutral, self.default_index)
		mapper.set_default_index(self.default_index)

		return mapper, dict(accumulator), len(mapper)

	def map_caption2index(self, caption):
		token2index = self.vocab_mapper.get_stoi()
		zeros = th.tensor([ token2index[self.neutral] ] * self.max_len)
		sequence = th.tensor([ token2index[tok] for tok in caption ])
		padded_sequences = pad_sequence([zeros, sequence], batch_first=True)
		return padded_sequences[:, :self.max_len][1]  # ignore the zeros entrie
	
	def map_index2caption(self, index):
		index2token = self.vocab_mapper.get_itos()
		return ' '.join([ index2token[idx] for idx in index if idx != self.default_index ])

	def get_caption(self, idx):
		crr_filename = self.filenames[idx]
		array_of_captions = self.captions_mapper[crr_filename]
		picked_idx = np.random.randint(len(array_of_captions))
		selected_caption = array_of_captions[picked_idx]
		return ' '.join(selected_caption)

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		crr_filename = self.filenames[idx]
		array_of_captions = self.captions_mapper[crr_filename]
		picked_idx = np.random.randint(len(array_of_captions))
		selected_caption = array_of_captions[picked_idx]
		
		seq_idx = self.map_caption2index(selected_caption)
		seq_len = (seq_idx != 0).sum().item()
		image = self.read_image(crr_filename)
		return image, seq_idx, seq_len

if __name__ == '__main__':
	source = DATAHOLDER('storage', shape=(256, 256))
	print(len(source))


	for i in range(7000, 7024):
		img, cap, lng = source[i]	
		img = (img * 0.5) + 0.5 
		txt = source.get_caption(i)
		print(txt)
		img = th2cv(img)
		cap = caption2image(txt)
		res = np.vstack([cap, img])
		cv2.imshow('###', res)
		cv2.waitKey(0)
