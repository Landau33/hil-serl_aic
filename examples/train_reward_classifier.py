import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer(
    "steps_per_epoch",
    100,
    "Number of gradient updates per epoch. The previous script only used 1, which is too noisy.",
)
flags.DEFINE_float("learning_rate", 1e-4, "Classifier learning rate.")
flags.DEFINE_string(
    "dataset_dir",
    None,
    "Directory containing classifier_data pickles. Defaults to ./classifier_data under cwd.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Directory to save classifier checkpoints. Defaults to ./classifier_ckpt under cwd.",
)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    dataset_dir = os.path.abspath(FLAGS.dataset_dir or os.path.join(os.getcwd(), "classifier_data"))
    checkpoint_dir = os.path.abspath(FLAGS.checkpoint_dir or os.path.join(os.getcwd(), "classifier_ckpt"))

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(dataset_dir, "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(dataset_dir, "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"dataset_dir: {dataset_dir}")
    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")
    if len(pos_buffer) == 0 or len(neg_buffer) == 0:
        raise ValueError(
            f"Dataset is empty or one-sided. success={len(pos_buffer)} failure={len(neg_buffer)} "
            f"under {dataset_dir}"
        )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   learning_rate=FLAGS.learning_rate,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        epoch_losses = []
        epoch_accuracies = []
        for _ in range(FLAGS.steps_per_epoch):
            # Sample equal number of positive and negative examples
            pos_sample = next(pos_iterator)
            neg_sample = next(neg_iterator)
            # Merge and create labels
            batch = concat_batches(
                pos_sample, neg_sample, axis=0
            )
            rng, key = jax.random.split(rng)
            obs = data_augmentation_fn(key, batch["observations"])
            batch = batch.copy(
                add_or_replace={
                    "observations": obs,
                    "labels": batch["labels"][..., None],
                }
            )

            rng, key = jax.random.split(rng)
            classifier, train_loss, train_accuracy = train_step(classifier, batch, key)
            epoch_losses.append(float(train_loss))
            epoch_accuracies.append(float(train_accuracy))

        print(
            "Epoch: "
            f"{epoch+1}, Train Loss: {np.mean(epoch_losses):.4f}, "
            f"Train Accuracy: {np.mean(epoch_accuracies):.4f}"
        )

    checkpoints.save_checkpoint(
        checkpoint_dir,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)
