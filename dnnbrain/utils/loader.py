import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImgLoader():
	def __init__(self):
		"""
		"""
		pass
		
	def gen_dataloader(self, imgpath, imgcropsize, transform = None, batch_size = 8, shuffle=True, num_workers=1):
		"""
		Generate dataloader from image path
		
		Parameters:
		------------
		imgpath[str]: path of stimuli picture
		imgresize[int/list]: resize images to make it suitable to input of a specific network
		transform[transform.Compose]: transformation ways
		batch_size[int]: batch size
		shuffle[bool]: shuffle images or not
		num_workers[int]: cpu workers used in model trainning
		"""
		if transform is None:
			transform = transforms.Compose([
							transforms.Resize(imgcropsize),
							transforms.ToTensor()
										])
		pak_datasets = datasets.ImageFolder(imgpath, transform)
		dataloader = DataLoader(pak_datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
		return dataloader