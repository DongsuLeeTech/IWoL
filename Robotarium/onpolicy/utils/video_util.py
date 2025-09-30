from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb

import pygame

from PIL import Image
import seaborn as sns
import io
import math

def record_video(label, renders=None, n_cols=None, skip_frames=1):
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return save_video(label, renders, n_cols=n_cols)

def save_video(label, tensor, fps=15, n_cols=1):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t

    if isinstance(tensor, pygame.Surface):
        # Convert the Surface object to numpy array
        tensor = [surface_to_tensor(t) for t in tensor]
    else:
        # If it's already a numpy array, we directly proceed
        tensor = np.array(tensor)

    # Prepare video tensor
    tensor = prepare_video(tensor, n_cols)
    tensor = _to_uint8(tensor)

    # tensor: (t, h, c, w) -> (t, c, h, w) for wandb
    print(tensor.shape)
    tensor = tensor.transpose(0, 2, 1, 3)
    print(tensor.shape)
    return wandb.Video(tensor, fps=fps, format='mp4')

def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape
    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    if v.shape[0] == 1:
        n_cols = 1
        n_rows = 1
    else:
        if n_cols is None:
            if v.shape[0] <= 4:
                n_cols = 2
            elif v.shape[0] <= 9:
                n_cols = 3
            elif v.shape[0] <= 16:
                n_cols = 4
            else:
                n_cols = 6

        if v.shape[0] % n_cols != 0:
            len_addition = n_cols - v.shape[0] % n_cols
            v = np.concatenate(
                (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
        n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))
    return v

def plot_and_log_communication_block(message, filename="communication_output.png"):
    """
    output: numpy array of shape (timesteps, num_agents, output_size)
    filename: name of the file to save the plot
    """
    timesteps, num_agents, output_size = message.shape

    # Create a figure for each agent
    fig, axs = plt.subplots(num_agents, 1, figsize=(10, num_agents * 3))

    # Iterate over each agent and plot its communication vector over time
    for agent_id in range(num_agents):
        # Take the average over the output size for simplification,
        # or you can plot specific dimensions if needed
        avg_output = np.mean(message[:, agent_id, :], axis=1)

        # Plot line for the current agent
        axs[agent_id].plot(np.arange(timesteps), avg_output, label=f"Agent {agent_id + 1}")
        axs[agent_id].set_title(f"Agent {agent_id + 1} Communication Output")
        axs[agent_id].set_xlabel("Timestep")
        axs[agent_id].set_ylabel("Output (averaged)")
        axs[agent_id].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(filename)
    print(f"PNG saved as {filename}")

    return wandb.Image(filename)

def save_gif_with_attention(attention_weights, slicin_idx = 1, filename="attention_weights.gif"):
    attention_weights = attention_weights[::slicin_idx]
    images = []
    num_timesteps, batch_size, num_heads, num_agents, _ = attention_weights.shape

    # Iterate through each timestep
    for t in range(num_timesteps):
        # Calculate rows and columns for subplots based on the number of heads
        cols = math.ceil(np.sqrt(num_heads))
        rows = math.ceil(num_heads / cols)

        # Create a figure with subplots for each head at this timestep
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axs = axs.flatten()  # Flatten the array for easy iteration

        # For each head, plot the attention weights in a subplot
        for h in range(num_heads):
            attn = attention_weights[t, 0, h]  # Use the first batch (B=0)

            sns.heatmap(attn, annot=True, cmap="viridis", cbar=True, ax=axs[h],
                        square=True, cbar_kws={'shrink': .8})
            axs[h].set_title(f"Head {h + 1} at Timestep {slicin_idx * (t + 1)}")
            axs[h].set_xlabel("Key Agents")
            axs[h].set_ylabel("Query Agents")

        # Remove empty subplots if any
        for h in range(num_heads, len(axs)):
            fig.delaxes(axs[h])

        # Adjust layout
        fig.suptitle(f"Timestep {slicin_idx * (t + 1)}", fontsize=16)
        plt.tight_layout()

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)

        # Open the image with Pillow and append to the image list
        img = Image.open(buf)
        images.append(img)
        plt.close(fig)

    # Save all frames as a GIF using Pillow
    images[0].save(filename, save_all=True, append_images=images[1:], duration=300, loop=0)
    print(f"GIF saved as {filename}")

    return wandb.Video(filename, format='gif')

def save_gif_with_graph(graph, slicin_idx = 1, filename="graphs.gif"):
    graph = graph[::slicin_idx]
    images = []
    num_timesteps, batch_size, num_agents, _ = graph.shape

    for t in range(num_timesteps):
        fig, ax = plt.subplots(figsize=(3, 3))

        tmp = graph[t, 0]

        sns.heatmap(tmp, annot=True, cmap="viridis", cbar=True, ax=ax,
                    square=True, cbar_kws={'shrink': .8})
        ax.set_title(f"Timestep {slicin_idx * (t + 1)}")
        ax.set_xlabel("Key Agents")
        ax.set_ylabel("Query Agents")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        buf.seek(0)

        img = Image.open(buf)
        images.append(img)
        plt.close(fig)

    images[0].save(filename, save_all=True, append_images=images[1:], duration=300, loop=0)
    print(f"GIF saved as {filename}")

    return wandb.Video(filename, format='gif')

