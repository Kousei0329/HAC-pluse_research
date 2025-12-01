# from https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
from torch import nn


class MaskedConv1d(nn.Conv1d):
	'''
	Implementation of Masked 1D convolution for point cloud data.
	Adapted from the PixelCNN paper for sequential/point cloud processing.

	Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
	Advances in neural information processing systems. 2016.
	https://arxiv.org/pdf/1606.05328.pdf
	'''
	def __init__(self, mask_type, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert mask_type in ('A', 'B')
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kW = self.weight.size()  # [out_channels, in_channels, kernel_size]
		self.mask.fill_(1)
		# Mask future positions in the sequence
		self.mask[:, :, kW // 2 + (mask_type == 'B'):] = 0

	def forward(self, x):
		# x: [N, C] for point clouds -> need to reshape to [B, C, L]
		if x.dim() == 2:
			# Reshape [N, C] -> [1, C, N] for Conv1d
			x = x.t().unsqueeze(0)
			self.weight.data *= self.mask
			out = super(MaskedConv1d, self).forward(x)
			# Reshape back [1, C_out, N] -> [N, C_out]
			return out.squeeze(0).t()
		else:
			# Standard Conv1d input [B, C, L]
			self.weight.data *= self.mask
			return super(MaskedConv1d, self).forward(x)


# Keep backward compatibility
class MaskedConv2d(nn.Conv2d):
	'''
	Implementation of the Masked convolution from the paper
	Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." Advances in neural information processing systems. 2016.
	https://arxiv.org/pdf/1606.05328.pdf
	'''
	def __init__(self, mask_type, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert mask_type in ('A', 'B')
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(1)
		self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
		self.mask[:, :, kH // 2 + 1:] = 0

	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv2d, self).forward(x)