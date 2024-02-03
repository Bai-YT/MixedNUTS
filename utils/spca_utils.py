import os, re
import pickle
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import numpy as np


def visualize_obj(array_3d, ss, cc, pp, save_name=''):
    # Directory to store individual frames
    os.makedirs('results/frames', exist_ok=True)

    # Create and save each slice as an image
    filenames = []
    if len(cc) == 1:  # Create s-p heatmaps
        plot_img = plt.imshow(
            np.log(array_3d[:, 0, :]),
            vmin=np.log(array_3d.min()), vmax=np.log(array_3d.max()),
            extent=[pp.min(), pp.max(), np.log(ss.min()), np.log(ss.max())],
            cmap='hot', interpolation='nearest', aspect='auto', origin='lower'
        )
        y_ticks_loc = np.log(ss)
        plt.yticks(y_ticks_loc, np.round(ss, 2))
        plt.xticks(np.round(pp, 2))
        plt.ylabel('s'); plt.xlabel('p')

        cbar = plt.colorbar(plot_img)
        cbar_ticks = cbar.get_ticks()
        cbar.set_ticklabels(np.round(np.exp(cbar_ticks), 2))

        filename = f"results/frames/cmap.png"
        filenames.append(filename)
        plt.tight_layout(); plt.savefig(filename); plt.close()

    else:  # Create s-c heatmaps for each p value
        for pid, p in enumerate(pp):
            plot_img = plt.imshow(
                np.log(array_3d[:, :, pid]),
                vmin=np.log(array_3d.min()), vmax=np.log(array_3d.max()),
                extent=[cc.min(), cc.max(), np.log(ss.min()), np.log(ss.max())],
                cmap='hot', interpolation='nearest', aspect='auto', origin='lower'
            )
            y_ticks_loc = np.log(ss)
            plt.yticks(y_ticks_loc, np.round(ss, 2))
            plt.xticks(np.round(cc, 2))
            plt.ylabel('s'); plt.xlabel('c')
            plt.title(f"Slice at p={p:.3f}")

            cbar = plt.colorbar(plot_img)
            cbar_ticks = cbar.get_ticks()
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(np.round(np.exp(cbar_ticks), 2))

            filename = f"results/frames/slice_{p}.png"
            filenames.append(filename)
            plt.tight_layout(); plt.savefig(filename); plt.close()

    # Assemble images into a video
    video_name = f'results/{save_name}_visualize_video.mp4'
    with imageio.get_writer(video_name, mode='I', fps=3) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)


def get_margin_file(
    custom_file_path: str, dataset_name: str, rob_model_name: str, image_type: str
) -> str:
    # Regular expression to match files that end with 'examples.pt'
    def extract_info(filename):
        match = re.search(r"(\d+)examples\.pt", filename)
        if match and image_type in filename:
            return int(match.group(1))
        return 0

    # List all files and filter out the one with the most examples
    if custom_file_path is not None:
        # Use user-specified file path
        selected_file_path = custom_file_path
    else:
        # Use default file path
        folder_path = f"results/{dataset_name}/{rob_model_name}/{image_type}"
        files = os.listdir(folder_path)
        max_examples = 0
        selected_file_path = None

        for file in files:
            examples = extract_info(file)
            if examples > max_examples:
                max_examples = examples
                selected_file_path = file[:-3]  # Remove the .pt suffix

    print(f"Loading margin info from {folder_path}/{selected_file_path}.pt...")
    with open(f"{folder_path}/{selected_file_path}.pt", 'rb') as margin_file:
        return pickle.load(margin_file)
