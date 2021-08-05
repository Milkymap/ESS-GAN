import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np
import operator as op 
import itertools as it, functools as ft 

class DOWNBLOCK(nn.Module):
	def __init__(self, icn, ocn, down=True):
		super(DOWNBLOCK, self).__init__()
		self.head = nn.Conv2d(icn, ocn, 1, 1, 0) if icn != ocn else nn.Identity()
		self.down = nn.AvgPool2d(2) if down else nn.Identity()
		self.body = nn.Sequential(
			nn.Conv2d(icn, ocn, 4, 2, 1, bias=False), 
			nn.LeakyReLU(0.2),
			nn.Conv2d(ocn, ocn, 3, 1, 1, bias=False), 
			nn.LeakyReLU(0.2)
		)
		self.parm = nn.Parameter(th.zeros(1))

	def forward(self, X):
		return self.down(self.head(X)) + self.parm * self.body(X)  

class CONCATBLOCK(nn.Module):
	def __init__(self, icn, ocn, tdf):
		super(CONCATBLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(icn + tdf, ocn, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2),
			nn.Conv2d(ocn, 1, 4, 1, 0, bias=False)
		)

	def forward(self, X, T):
		_, _, H, W = X.shape
		S = T[:, :, None, None].repeat(1, 1, H, W)
		XS = th.cat((X, S), dim=1)
		return th.squeeze(self.body(XS))

class DISCRIMINATOR(nn.Module):
	def __init__(self, icn, ndf, tdf, min_idx, nb_dblocks):
		super(DISCRIMINATOR, self).__init__()
		self.head = nn.Conv2d(icn, ndf, 3, 1, 1) 
		self.body = nn.Sequential(*
			[ 
				DOWNBLOCK(ndf * 2 ** np.minimum(min_idx, idx), ndf * 2 ** np.minimum(min_idx, idx + 1) ) 
				for idx in range(nb_dblocks) 
			]
		)
		self.tail = CONCATBLOCK(ndf * 2 ** min_idx, ndf * 2 ,tdf) 

	def forward(self, X, T):
		Y = self.body(self.head(X))
		return self.tail(Y, T)

if __name__ == '__main__':
	X = th.randn((2, 3, 64, 64)).requires_grad_()
	T = th.randn((2, 256)).requires_grad_()

	D = DISCRIMINATOR(3, 64, 256, 4, 4)
	Q = D(X, T)
	print(Q)