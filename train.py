import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
from utils import save_model
from utils import visualize
import params

# Source : 0, Target :1
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader


def source_only(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name):
    print("Source-only training")
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))

        encoder = encoder.train()
        classifier = classifier.train()
        discriminator = discriminator.train()

        classifier_criterion = nn.CrossEntropyLoss().cuda()

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(classifier.parameters()),
            lr=0.001, momentum=0.9)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), class_loss.item()))

        if (epoch + 1) % 10 == 0:
            test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='source_only')
    save_model(encoder, classifier, discriminator, 'source', save_name)
