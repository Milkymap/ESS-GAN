import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision as tv 


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import inception_v3 as IV3 

class RNN_ENCODER(nn.Module):
	def __init__(self, h_size, n_layers, p, n_emb, e_dim, p_idx):
		super(RNN_ENCODER, self).__init__()
		self.hidden_size = h_size
		self.numb_layers = n_layers
		
		self.drop = nn.Dropout(p) if p > 0.0 else nn.Identity()
		self.head = nn.Embedding(num_embeddings=n_emb, embedding_dim=e_dim, padding_idx=p_idx)
		self.body = nn.LSTM(input_size=e_dim, hidden_size=h_size, num_layers=n_layers, batch_first=True, bidirectional=True)
		
		self.head.weight.data.uniform_(-0.1, 0.1)
	
	def forward(self, T, seq_length):
		batch_size, max_length  = T.shape
		hidden_cell_0 = (
			th.zeros(2 * self.numb_layers, batch_size, self.hidden_size).to(next(self.parameters()).device),
			th.zeros(2 * self.numb_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
		)

		embedded = self.drop(self.head(T))  # 2D => 3D
		packed_embedded = pack_padded_sequence(embedded, seq_length, batch_first=True)
		rnn_packed_response, (hidden_1, _ ) = self.body(packed_embedded, hidden_cell_0)
		rnn_padded_response, _ = pad_packed_sequence(rnn_packed_response, batch_first=True, total_length=max_length)
		words_embedding = rnn_padded_response.transpose(1, 2)       #BxHxS
		global_sentence_embedding = th.cat(tuple(hidden_1), dim=1)  #BxH 

		return words_embedding, global_sentence_embedding.transpose(0, 1) 

class CNN_ENCODER(nn.Module):
	def __init__(self, nef):
		super(CNN_ENCODER, self).__init__()
		model = IV3(pretrained=True)
		for prm in model.parameters():
			prm.requires_grad = False

		layers = list(model.children())

		self.init = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)
		self.head = nn.Sequential(*layers[:15])
		self.body = nn.Sequential(*layers[16:20])
		self.cnn_ = nn.Conv2d(768, nef, 1, 1)
		self.mlp_ = nn.Linear(2048, nef)

		self.cnn_.weight.data.uniform_(-0.1, 0.1)
		self.mlp_.weight.data.uniform_(-0.1, 0.1)

	def forward(self, X0):
		X1 = self.head(self.init(X0))
		X2 = self.body(X1)
		return th.flatten(self.cnn_(X1), start_dim=2), self.mlp_(X2.view(X2.size(0), -1)).transpose(0, 1)

class DAMSM(nn.Module):

	"""
		B, _, _ = W.shape
		S = th.einsum('ijk,ijn->ikn', W, I)
		S = th.softmax(gamma_1 * th.softmax(S, dim=1), dim=2)
		C = th.einsum('ijk,->imk->imj', S, I)
		R = F.cosine_similarity(C, W, dim=1)
		M = th.sum(th.exp(gamma_2 * R), dim=1) ** (1 / gamma_2)
		W = th.reshape(M, (B, B))
	"""

	def __init__(self, vocab_size, common_space_dim=256, e_dim=300):
		super(DAMSM, self).__init__()
		self.rnn_network = RNN_ENCODER(h_size=common_space_dim // 2, n_layers=1, p=0.1, n_emb=vocab_size, e_dim=e_dim, p_idx=0)
		self.cnn_network = CNN_ENCODER(common_space_dim)

	def encode_seq(self, seq, seq_length):
		return self.rnn_network(seq, seq_length)

	def encode_img(self, img):
		return self.cnn_network(img)

	def forward(self, img, seq, seq_length):
		words_features, sentence_features = self.encode_seq(seq, seq_length)
		local_image_features, global_image_features = self.encode_img(img)
		return words_features, sentence_features, local_image_features, global_image_features

def local_match_probabilities(words, image, gamma_1=5, gamma_2=5, gamma_3=10):
	N, _, _ = words.shape
	words_ = th.tile(words, (N, 1, 1))
	image_ = th.repeat_interleave(image, N, dim=0)

	sim_matrix = th.einsum('ijk,ijn->ikn', words_, image_)              
	normalized_matrix = th.softmax(sim_matrix, dim=1)                     
	attention_coeficients = th.softmax(gamma_1 * normalized_matrix, dim=2)
	regions_context = th.einsum('ijk,imk->imj', attention_coeficients, image_)            

	words_regions_sim = F.cosine_similarity(regions_context, words_, dim=1)
	image_description_score = th.log(th.sum(th.exp(gamma_2 * words_regions_sim), dim=1))
	batch_image_words_matching_score = th.reshape(image_description_score, (N, N))
	
	prob_d_given_q = gamma_3 * batch_image_words_matching_score
	prob_q_given_d = prob_d_given_q.transpose(0, 1)
	return prob_d_given_q, prob_q_given_d


def global_match_probabilities(sentences, features, gamma_3=10):
	_, N = sentences.shape  
	sentences_ = th.tile(sentences, (1, N))
	features_ = th.repeat_interleave(features, N, dim=1)

	image_description_score = F.cosine_similarity(sentences_, features_, dim=0)
	batch_features_sentences_matching_score = th.reshape(image_description_score, (N, N))
	
	prob_d_given_q = gamma_3 * batch_features_sentences_matching_score
	prob_q_given_d = prob_d_given_q.transpose(0, 1)
	return prob_d_given_q, prob_q_given_d 


if __name__ == '__main__':
	"""
	T = th.randint(10, 90, size=(5, 15))
	I = th.randn((5, 3, 256, 256))
	
	D = DAMSM(vocab_size=100)
	W, E, L, G = D(I, T, [15] * 5)
	

	P0, P1 = D.local_match_probabilities(W, L)
	print(P0)
	print(P1)
	Q0, Q1 = D.global_match_probabilities(E, G)
	print(Q0)
	print(Q1)
	"""
	