import argparse
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
import optax
import torch
import torch.utils.checkpoint
import transformers
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from jax.experimental.compilation_cache import compilation_cache as cc
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
from diffusers.utils import check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0.dev0")

# Cache compiled models across invocations of this script.
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

logger = logging.getLogger(__name__)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main(
    pretrained_model_name_or_path: Optional[str],
    pretrained_vae_name_or_path: Optional[str],
    revision: Optional[str],
    tokenizer_name: Optional[str],
    instance_data_dir: Optional[str],
    class_data_dir: Optional[str],
    instance_prompt: Optional[str],
    class_prompt: Optional[str],
    with_prior_preservation: bool,
    prior_loss_weight: float,
    num_class_images: int,
    output_dir: str,
    save_steps: Optional[int],
    seed: int,
    resolution: int,
    center_crop: bool,
    train_text_encoder: bool,
    train_batch_size: int,
    sample_batch_size: int,
    num_train_epochs: int,
    max_train_steps: Optional[int],
    learning_rate: float,
    scale_lr: bool,
    adam_beta1: float,
    adam_beta2: float,
    adam_weight_decay: float,
    adam_epsilon: float,
    max_grad_norm: float,
    push_to_hub: bool,
    hub_token: Optional[str],
    hub_model_id: Optional[str],
    logging_dir: str,
    mixed_precision: str,
    local_rank: int,
):
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != local_rank:
        local_rank = env_local_rank

    if instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if with_prior_preservation:
        if class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    rng = jax.random.PRNGKey(seed)

    if with_prior_preservation:
        class_images_dir = Path(class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, safety_checker=None, revision=revision
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(class_prompt, num_new_images)
            total_sample_batch_size = sample_batch_size * jax.local_device_count()
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=total_sample_batch_size)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not jax.process_index() == 0
            ):
                prompt_ids = pipeline.prepare_inputs(example["prompt"])
                prompt_ids = shard(prompt_ids)
                p_params = jax_utils.replicate(params)
                rng = jax.random.split(rng)[0]
                sample_rng = jax.random.split(rng, jax.device_count())
                images = pipeline(prompt_ids, p_params, sample_rng, jit=True).images
                images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
                images = pipeline.numpy_to_pil(np.array(images))

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline

    # Handle the repository creation
    if jax.process_index() == 0:
        if push_to_hub:
            if hub_model_id is None:
                repo_name = get_full_repo_name(Path(output_dir).name, token=hub_token)
            else:
                repo_name = hub_model_id
            create_repo(repo_name, exist_ok=True, token=hub_token)
            repo = Repository(output_dir, clone_from=repo_name, token=hub_token)

            with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    elif pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
        )
    else:
        raise NotImplementedError("No tokenizer specified!")

    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        class_data_root=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt,
        class_num=num_class_images,
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch

    total_train_batch_size = train_batch_size * jax.local_device_count()
    if len(train_dataset) < total_train_batch_size:
        raise ValueError(
            f"Training batch size is {total_train_batch_size}, but your dataset only contains"
            f" {len(train_dataset)} images. Please, use a larger dataset or reduce the effective batch size. Note that"
            f" there are {jax.local_device_count()} parallel devices, so your batch size can't be smaller than that."
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=total_train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    weight_dtype = jnp.float32
    if mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    if pretrained_vae_name_or_path:
        # TODO(patil-suraj): Upload flax weights for the VAE
        vae_arg, vae_kwargs = (pretrained_vae_name_or_path, {"from_pt": True})
    else:
        vae_arg, vae_kwargs = (pretrained_model_name_or_path, {"subfolder": "vae", "revision": revision})

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", dtype=weight_dtype, revision=revision
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        vae_arg,
        dtype=weight_dtype,
        **vae_kwargs,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", dtype=weight_dtype, revision=revision
    )

    # Optimization
    if scale_lr:
        learning_rate = learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=adam_beta1,
        b2=adam_beta2,
        eps=adam_epsilon,
        weight_decay=adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adamw,
    )

    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    text_encoder_state = train_state.TrainState.create(
        apply_fn=text_encoder.__call__, params=text_encoder.params, tx=optimizer
    )

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Initialize our training
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        if train_text_encoder:
            params = {"text_encoder": text_encoder_state.params, "unet": unet_state.params}
        else:
            params = {"unet": unet_state.params}

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            if train_text_encoder:
                encoder_hidden_states = text_encoder_state.apply_fn(
                    batch["input_ids"], params=params["text_encoder"], dropout_rng=dropout_rng, train=True
                )[0]
            else:
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], params=text_encoder_state.params, train=False
                )[0]

            # Predict the noise residual
            model_pred = unet.apply(
                {"params": params["unet"]}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if with_prior_preservation:
                # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
                target, target_prior = jnp.split(target, 2, axis=0)

                # Compute instance loss
                loss = (target - model_pred) ** 2
                loss = loss.mean()

                # Compute prior loss
                prior_loss = (target_prior - model_pred_prior) ** 2
                prior_loss = prior_loss.mean()

                # Add the prior loss to the instance loss.
                loss = loss + prior_loss_weight * prior_loss
            else:
                loss = (target - model_pred) ** 2
                loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params)
        grad = jax.lax.pmean(grad, "batch")

        new_unet_state = unet_state.apply_gradients(grads=grad["unet"])
        if train_text_encoder:
            new_text_encoder_state = text_encoder_state.apply_gradients(grads=grad["text_encoder"])
        else:
            new_text_encoder_state = text_encoder_state

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_unet_state, new_text_encoder_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0, 1))

    # Replicate the train state on each device
    unet_state = jax_utils.replicate(unet_state)
    text_encoder_state = jax_utils.replicate(text_encoder_state)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    def checkpoint(step=None):
        # Create the pipeline using the trained modules and save it.
        scheduler, _ = FlaxPNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker", from_pt=True
        )
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        )

        outdir = os.path.join(output_dir, str(step)) if step else output_dir
        pipeline.save_pretrained(
            outdir,
            params={
                "text_encoder": get_params_to_save(text_encoder_state.params),
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(unet_state.params),
                "safety_checker": safety_checker.params,
            },
        )

        if push_to_hub:
            message = f"checkpoint-{step}" if step is not None else "End of training"
            repo.push_to_hub(commit_message=message, blocking=False, auto_lfs_prune=True)

    global_step = 0

    epochs = tqdm(range(num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = shard(batch)
            unet_state, text_encoder_state, train_metric, train_rngs = p_train_step(
                unet_state, text_encoder_state, vae_params, batch, train_rngs
            )
            train_metrics.append(train_metric)

            train_step_progress_bar.update(jax.local_device_count())

            global_step += 1
            if jax.process_index() == 0 and save_steps and global_step % save_steps == 0:
                checkpoint(global_step)
            if global_step >= max_train_steps:
                break

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{num_train_epochs} | Loss: {train_metric['loss']})")

    if jax.process_index() == 0:
        checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_dreambooth_flax.yaml")
    args = parser.parse_args()
    configs = OmegaConf.load(args.config)

    main(**configs)
