import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.task, librl.train.train_loop, librl.train.cc
import pytest

from . import * 

#########
# Tests #
#########

# Train an MLP network generator using VPG, PGB, and PPO
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA.")
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_mlp(mnist_dataset, train_fn):
    mnist_dataset.hypers['device'] = 'cuda'
    train_fn(*mlp_helper(mnist_dataset))

# Train an CNN network generator using VPG, PGB, and PPO.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA.")
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_cnn(mnist_dataset, train_fn):
    mnist_dataset.hypers['device'] = 'cuda'
    train_fn(*cnn_helper(mnist_dataset))

# Train an MLP+CNN network generator using VPG, PGB, and PPO.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA.")
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_all(mnist_dataset, train_fn):
    mnist_dataset.hypers['device'] = 'cuda'
    train_fn(*joint_helper(mnist_dataset))