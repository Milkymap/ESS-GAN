import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torchvision.models import vgg16

from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 
  
from models.generator import GENERATOR
from models.discriminator import DISCRIMINATOR
from models.damsm import *

from os import path, mkdir  

@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int, default=600)
@click.option('--bt_size', help='batch size', type=int, default=4)
@click.option('--noise_dim', help='dimension of the noise vector Z', default=100)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
@click.option('--images_store', help='generated images will be stored in this directory', default='images_store')
def main_loop(storage, nb_epochs, bt_size, noise_dim, pretrained_model, images_store):
	# intitialization : device, dataset and dataloader 
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	source = DATAHOLDER(path_to_storage=storage, max_len=18, neutral='<###>', shape=(256, 256), nb_items=1024)
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	if not path.isdir(images_store):
		mkdir(images_store)

	# create networks : dams, generator and discriminator
	if pretrained_model != '' and path.isfile(pretrained_model):
		encoder_network = th.load(pretrained_model, map_location=device)
		for p in encoder_network.parameters():
			p.requires_grad = False 
		encoder_network.eval()
		logger.debug('The pretrained DAMSM was loaded')

	generator_network = GENERATOR(noise_dim=noise_dim, tsp=256).to(device) 
	discriminator_network = DISCRIMINATOR(icn=3, ndf=64, tdf=256, min_idx=4, nb_dblocks=6).to(device)

	generator_network.train()
	discriminator_network.train()
	logger.debug('Generator and Discriminator were created')

	extractor = vgg16(pretrained=True)
	extractor_network = nn.Sequential(*list(extractor.features)).eval().to(device)
	extractor_network.eval()

	for p in extractor_network.parameters():
		p.requires_grad = False 

	# define hyparameters
	lambda_dams = 5e-2
	lambda_ragan = 5e-3
	lambda_content = 1e-2
	
	# define solvers and criterions
	 
	perceptual_criterion = nn.MSELoss().to(device)
	content_criterion = nn.L1Loss().to(device) 
	ragan_criterion = nn.BCEWithLogitsLoss().to(device)

	generator_solver = optim.Adam(generator_network.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminator_solver = optim.Adam(discriminator_network.parameters(), lr=2e-4, betas=(0.5, 0.999)) 

	nb_images = 0 
	total_images = len(source)

	# main training loop  
	for epoch_counter in range(nb_epochs):
		nb_images = 0
		for index, (real_images, captions, lengths) in enumerate(loader.loader):
			# size current batch
			batch_size = len(real_images)   
			nb_images = nb_images + batch_size 

			#define labels 
			real_labels = th.ones(batch_size).to(device)
			fake_labels = th.zeros(batch_size).to(device)

			# move data to target device : gpu or cpu 
			real_images = real_images.to(device)
			captions = captions.to(device)
			
			# caption encoding
			response = encoder_network.encode_seq(captions, lengths)
			words, sentences = list(map(lambda M: M.detach(), response))
			transposed_sentences = sentences.transpose(0, 1)

			#-----------------------------#
			# train generator network #
			#-----------------------------#

			# synthetize fake real_images
			noise = th.randn((batch_size, noise_dim)).to(device)
			fake_images, predicted_masks = generator_network(noise, transposed_sentences)

			fake_images_features = extractor_network(fake_images)
			real_images_features = extractor_network(real_images).detach()

			fake_images_critics = discriminator_network(fake_images, transposed_sentences)
			real_images_critics = discriminator_network(real_images, transposed_sentences).detach()

			# compute damsm loss 
			response = network.encode_img(fake_images)	
			local_features, global_features = list(map(lambda M: M.detach(), response)) 
			
			wq_prob, qw_prob = local_match_probabilities(words, local_features)
			sq_prob, qs_prob = global_match_probabilities(sentence, global_features)

			loss_w1 = criterion_damsm(wq_prob, damsm_labels) 
			loss_w2 = criterion_damsm(qw_prob, damsm_labels)
			loss_s1 = criterion_damsm(sq_prob, damsm_labels)
			loss_s2 = criterion_damsm(qs_prob, damsm_labels)

			generator_dams_loss = loss_w1 + loss_w2 + loss_s1 + loss_s2

			generator_ra_loss = ragan_criterion(fake_images_critics - th.mean(real_images_critics), real_labels)
			generator_content_loss = content_criterion(fake_images, real_images)
			generator_perceptual_loss = perceptual_criterion(fake_images_features, real_images_features)
			
			generator_error = generator_perceptual_loss + lambda_content * generator_content_loss + lambda_ragan * generator_ra_loss + lambda_dams * generator_dams_loss

			# backpropagate the error through the generator and dams network

			generator_solver.zero_grad()
			generator_error.backward()
			generator_solver.step()

			#-----------------------------#
			# train discriminator network #
			#-----------------------------#
			fake_images_critics = discriminator_network(fake_images.detach(), transposed_sentences)
			real_images_critics = discriminator_network(real_images, transposed_sentences)

			discriminator_real_loss = ragan_criterion(real_images_critics - th.mean(fake_images_critics), real_labels)
			discriminator_fake_loss = ragan_criterion(fake_images_critics - th.mean(real_images_critics), fake_labels)
			
			discriminator_error = (discriminator_real_loss + discriminator_fake_loss) / 2

			discriminator_solver.zero_grad()
			discriminator_error.backward()
			discriminator_solver.step()
			
			#---------------------------------------------#
			# debug some infos : epoch counter, loss value#
			#---------------------------------------------#
		
			message = (nb_images, total_images, epoch_counter, nb_epochs, index, generator_error.item(), discriminator_error.item())
			logger.debug('[%04d/%04d] | [%03d/%03d]:%05d | GLoss : %07.3f | DLoss : %07.3f' % message)
			
			if index % 2 == 0:
				descriptions = [ source.map_index2caption(seq) for seq in captions]
				output = snapshot(real_images.cpu(), fake_images.cpu(), descriptions, f'output epoch {epoch_counter:03d}', mean=[0.5], std=[0.5])
				cv2.imwrite(path.join(images_store, f'###_{epoch_counter:03d}_{index:03d}.jpg'), output)
				
		# temporary model states
		if epoch_counter % 100 == 0:
			th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')		
			th.save(discriminator_network, f'dump/discriminator_{epoch_counter:03d}.th')		
	
	th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')
	th.save(discriminator_network, f'dump/discriminator_{epoch_counter:03d}.th')		
	
	logger.success('End of training ...!')

if __name__ == '__main__':
	main_loop()


