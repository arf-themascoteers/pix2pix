from torchvision.utils import make_grid
import matplotlib.pyplot as plt


Normalization_Values = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def DeNormalize(tensor_of_image):
  return tensor_of_image * Normalization_Values[1][0] + Normalization_Values[0][0]


def print_images(image_tensor, num_images):
    images = DeNormalize(image_tensor)
    images = images.detach().cpu()
    image_grid = make_grid(images[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()