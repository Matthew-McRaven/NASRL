import gym, gym.spaces
import more_itertools
from more_itertools.more import sample
import numpy as np
import torch

import librl.replay
import librl.nn.core, librl.nn.classifier
import librl.task, librl.task.classification

# Given a set of classification task samples, train the task's classifer on the task data.
# This helper method assumes classifiers do not share components, and that we are
# only dealing in assigning a single label to each data element.
def test_gen_classifier(task):
    assert task.problem_type == librl.task.ProblemTypes.Classification
    for dataloader in task.train_data_iter:
        task.classifier.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            loss, selected = task.train_batch(task.classifier, task.criterion, data, target)
            #accuracy = torch.eq(selected, target).sum() / float(target.shape[0])
            #print(f"Batch {batch_idx} with loss {loss}")
    correct, total = 0,0
    for dataloader in task.validation_data_iter:
        task.classifier.eval()
        for batch_idx, (data, target) in enumerate(dataloader):
            loss, selected = task.validate_batch(task.classifier, task.criterion, data, target)
            #print(target, selected)
            correct += torch.eq(selected, target).sum() 
            total += float(target.shape[0])
    print(f"Accuracy of {correct/total}")
    return correct, total
        
       
def create_nn_from_def(input_dimension, conv_layers=None, linear_layers=None):
    assert not (conv_layers == None and linear_layers == None)
    module_list = []
    intermediate_dim=input_dimension
    if conv_layers:
        # Last (2) elements of input dim should be xy, 0'th should be # channels.
        module_list.append(librl.nn.core.cnn.ConvolutionalKernel(conv_layers, intermediate_dim[1:],intermediate_dim[0]))
        intermediate_dim = module_list[0].output_dimension
    if linear_layers: module_list.append(librl.nn.core.MLPKernel(intermediate_dim, linear_layers))
    return librl.nn.core.SequentialKernel(module_list)

