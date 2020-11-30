import unittest

import librl.nn.core, librl.nn.classifier
import librl.reward
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
import librl.utils
import librl.train.classification
import torch, torch.utils
import torchvision.datasets, torchvision.transforms


# Modified version of librl/tests/reco_test_label.
# Confirms that the image reco functionality of librl
# has been correctly configured inside nasrl.
class TestClassification(unittest.TestCase):

    def test_label_mnist(self):
        hypers = {}
        hypers['device'] = 'cpu'
        hypers['epochs'] = 2
        hypers['task_count'] = 1

        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),]) 
        # Load the MNIST training / validation datasets
        train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
        validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)

        class_kernel = librl.nn.core.MLPKernel((1, 28, 28), (200, 100))
        class_net = librl.nn.classifier.Classifier(class_kernel, 10)

        dist = librl.task.TaskDistribution()
        # Construct dataloaders from datasets
        t_loaders, v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
        # Construct a labelling task.
        dist.add_task(librl.task.Task.Definition(librl.task.ClassificationTask, classifier=class_net, criterion=torch.nn.CrossEntropyLoss(), train_data_iter=t_loaders, validation_data_iter=v_loaders))
        librl.train.train_loop.cls_trainer(hypers, dist, librl.train.classification.train_single_label_classifier)


if __name__ == "__main__":
    unittest.main()