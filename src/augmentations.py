import torch
from torchvision import transforms

class GaussianNoise:

	#add random noise to an image

	def __init__(self,standard_deviation = 0.1):
		self.standard_deviation = standard_deviation
	
	def __call__(self,image):
		noise =torch.randn_like(image) * self.standard_deviation

		noisy_image = image + noise

		return torch.clamp(noisy_image,0,1)

class TwoViews:
	#create two versions of the same image
	#since SimCLR needs two version of each image so that it can compare
	def __init__(self,transform):
		self.transform = transform
	def __call__(self,image):
		view1 = self.transform(image)
		view2 = self.transform(image)

		return view1,view2

class AugmentationFactory:
	#create the augmentation that we wnt to use in an experiment.

	def create(self,augmentation_name):
		if augmentation_name == "crop":

			transform = transforms.Compose([
				transforms.RandomResizedCrop(32),
				transforms.ToTensor()
			])
		elif augmentation_name == "rotation":
			transform = transforms.Compose([
				transforms.RandomRotation(20),
				transforms.ToTensor()
			])
		elif augmentation_name =="color_jitter":
			transform = transforms.Compose([
				transforms.ColorJitter(
					brightness =0.4,
					contrast = 0.4,
					saturation = 0.4
				),
				transforms.ToTensor()
			])
		elif augmentation_name =="gaussian_noise":
			transform = transforms.Compose([
				transforms.ToTensor(),
				GaussianNoise(0.1)
			])
		elif augmentation_name == "none":
			transform = transforms.Compose([
				transforms.ToTensor()
			])
		else:
			raise ValueError(
				"Unknown augmentation: " + 
				augmentation_name
			)
		return TwoViews(transform)

	def create_evaluation_transform(self):
		transform = transforms.Compose([
			transforms.ToTensor()
		])

		return transform