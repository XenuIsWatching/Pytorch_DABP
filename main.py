import torch
import train
import test
import mnist
import mnistm
import model
from utils import get_free_gpu, visualize, visualize_input

save_name = 'exp'


def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        train.source_only(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name)
        test.tester(encoder, classifier, discriminator, source_train_loader, target_train_loader, 'source')

        visualize_input()
        visualize(encoder, 'source', 'encoded')

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()
