from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from .customize_box_refine import merge_boxes_in_same_timestamp
from PIL import Image, ImageDraw, ImageFont
import imageio


def plot_box_on_single_image(image_folder, image_name, boxes, save_folder, labels, pvals, print_label=False,
                      box_color = 'r', box_thickness = 2):
    """
    Plots a bounding box on an image, optionally with a label, and saves it.

    Parameters:
        image_path (str): Path to the input image.
        box (tuple): Tuple of (x1, y1, x2, y2) coordinates for the bounding box.
        save_path (str): Path where the output image will be saved.
        label (str): The label to print.
        pval (float): The p-value associated with the label.
        print_label (bool): Whether to print the label and p-value.
    """
    image_path = image_folder + image_name
    if print_label and len(boxes) != len(pvals):
        print('Please double check if some pvals or box coordinates are missing')
    else:
        if not isinstance(labels, list):
            labels = [labels]
            for i in range(len(boxes) - 1):
                labels.append(labels[0])
    # Load image
    image = Image.open(image_path)
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(image)
    
    for i in range(len(boxes)):
        box, label, pval = boxes[i], labels[i], pvals[i]
        # Create a rectangle patch
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=box_thickness, edgecolor=box_color, facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Check if label and p-value should be printed
        if print_label and label is not None and pval is not None:
            label_text = f"{label}: {pval:.3f}"  # Formatting p-value in scientific notation
            # Position the text at the bottom of the box
            ax.text(x1, y2, label_text, fontsize=12, verticalalignment='top', color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    # Remove axes
    ax.axis('off')
    
    
    os.makedirs(save_folder, exist_ok=True)
    # Save the figure
    plt.savefig(save_folder + image_name, bbox_inches='tight', pad_inches=0, dpi = 300)
    plt.close()

def plot_box_on_whole_image(image_folder, temp_datetime, temp_dict, save_folder, labels = 'Rip', print_label=False, image_suffix = 'PW',
                      image_ending = 'jpg', image_height = 551, image_width = 551,
                      box_color = 'r', box_thickness = 2, save_name = ''):
    image_num = len(temp_dict)
    new_image = Image.new('RGB', (image_width * image_num, image_height))
    for k in temp_dict.keys():
        ind = int(k) - 1
        image_name = '%s-%s_%s.%s'%(temp_datetime, image_suffix, str(k), image_ending)
        temp_image = Image.open(image_folder + image_name)
        new_image.paste(temp_image, (ind * image_width, 0))
        
    boxes, pvals = merge_boxes_in_same_timestamp(temp_dict, image_width, sensitivity_thresh = 2)
    if print_label and len(boxes) != len(pvals):
        print('Please double check if some pvals or box coordinates are missing')
    else:
        if not isinstance(labels, list):
            labels = [labels]
            for i in range(len(boxes) - 1):
                labels.append(labels[0])
    
    # Create a figure and axes
    fig, ax = plt.subplots(dpi=3000)
    
    # Display the image
    ax.imshow(new_image)
    
    for i in range(len(boxes)):
        box, label, pval = boxes[i], labels[i], pvals[i]
        # Create a rectangle patch
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=box_thickness, edgecolor=box_color, facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Check if label and p-value should be printed
        if print_label and label is not None and pval is not None:
            label_text = f"{label}: {pval:.3f}"  # Formatting p-value in scientific notation
            # Position the text at the bottom of the box
            ax.text(x1, y2, label_text, fontsize=12, verticalalignment='top', color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    # Remove axes
    ax.axis('off')
    
    
    os.makedirs(save_folder, exist_ok=True)
    # Save the figure
    if not save_name:
        save_name = '%s-%s_wo_refine.%s'%(temp_datetime, image_suffix, image_ending)
    plt.savefig(save_folder + save_name, bbox_inches='tight', pad_inches=0, dpi = 3000)
    plt.close()

def plot_box_on_merged_image(image_folder, temp_datetime, key_list, temp_dict, save_folder, labels = 'Rip', print_label=False, image_suffix = 'PW',
                      image_ending = 'jpg', image_height = 551, image_width = 551,
                      box_color = 'r', box_thickness = 2, save_name = ''):
    image_num = len(key_list)
    new_image = Image.new('RGB', (image_width * image_num, image_height))
    for k in key_list:
        ind = int(k) - 1
        image_name = '%s-%s_%s.%s'%(temp_datetime, image_suffix, str(k), image_ending)
        temp_image = Image.open(image_folder + image_name)
        temp_image = temp_image.resize((image_width, image_height), Image.ANTIALIAS)
        new_image.paste(temp_image, (ind * image_width, 0))
    boxes, pvals = temp_dict.get('bboxes'), temp_dict.get('pvals')    
    if print_label and len(boxes) != len(pvals):
        print('Please double check if some pvals or box coordinates are missing')
    else:
        if not isinstance(labels, list):
            labels = [labels]
            for i in range(len(boxes) - 1):
                labels.append(labels[0])
    
    # Create a figure and axes
    fig, ax = plt.subplots(dpi=3000)
    
    # Display the image
    ax.imshow(new_image)
    
    for i in range(len(boxes)):
        box, label, pval = boxes[i], labels[i], pvals[i]
        # Create a rectangle patch
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=box_thickness, edgecolor=box_color, facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Check if label and p-value should be printed
        if print_label and label is not None and pval is not None:
            label_text = f"{label}: {pval:.3f}"  # Formatting p-value in scientific notation
            # Position the text at the bottom of the box
            ax.text(x1, y2, label_text, fontsize=12, verticalalignment='top', color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    # Remove axes
    ax.axis('off')
    
    
    os.makedirs(save_folder, exist_ok=True)
    # Save the figure
    if not save_name:
        save_name = '%s-%s_w_refine.%s'%(temp_datetime, image_suffix, image_ending)
    plt.savefig(save_folder + save_name, bbox_inches='tight', pad_inches=0, dpi = 3000)
    plt.close()

def plot_merge_boxes_on_multiple_timestamps(image_folder, datetime_list, key_list, input_dict, save_folder, labels='Rip', print_label=False, 
                         image_suffix='PW', image_ending='jpg', image_height=551, image_width=551, box_color='r', 
                         box_thickness=2, save_name='', output_mode='images', annotate_datetime = False):
    """
    Function to plot boxes on merged images for each datetime, save as multiple images or a GIF.

    Parameters:
        image_folder (str): Directory containing the images.
        datetime_list (list): List of datetime strings.
        key_list (list): List of keys to identify which images to merge.
        temp_dict (dict): Dictionary containing 'bboxes' and 'pvals'.
        save_folder (str): Output directory to save the result.
        labels (str or list): Label to apply on each box.
        print_label (bool): Flag to determine whether to print the label and p-value.
        image_suffix (str): Suffix for the image filenames.
        image_ending (str): File extension for the images.
        image_height (int): Height of each image.
        image_width (int): Width of each image.
        box_color (str): Color of the box edges.
        box_thickness (int): Thickness of the box edges.
        save_name (str): Base name for saving the output file.
        output_mode (str): 'images' to save as multiple images or 'gif' to save as a single GIF.
    """
    os.makedirs(save_folder, exist_ok=True)
    images_for_gif = []  # List to store images if saving as GIF

    for temp_datetime in datetime_list:
        image_num = len(key_list)
        new_image = Image.new('RGB', (image_width * image_num, image_height))
        for k in key_list:
            ind = int(k) - 1
            image_name = f'{temp_datetime}-{image_suffix}_{str(k)}.{image_ending}'
            temp_image = Image.open(os.path.join(image_folder, image_name))
            new_image.paste(temp_image, (ind * image_width, 0))

        # Check label consistency with boxes and p-values
        temp_dict = input_dict.get(temp_datetime)
        boxes, pvals = temp_dict.get('bboxes'), temp_dict.get('pvals')
        if print_label and len(boxes) != len(pvals):
            print('Please double check if some pvals or box coordinates are missing')
        else:
            if not isinstance(labels, list):
                labels = [labels] * len(boxes)

        # Plotting
        fig, ax = plt.subplots(figsize=(image_width * image_num / 100, image_height / 100), dpi=300)
        ax.imshow(new_image)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=box_thickness, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
            if print_label:
                label_text = f"{labels[i]}: {pvals[i]:.3f}"
                ax.text(x1, y2, label_text, fontsize=12, verticalalignment='top', color='white', bbox=dict(facecolor='red', alpha=0.5))
        
        ax.text(0.5, 0.08, temp_datetime, transform=ax.transAxes, fontsize=24,
            horizontalalignment='center', verticalalignment='bottom',
            color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        if output_mode == 'images':
            # Save each figure individually
            individual_save_name = f'{temp_datetime}-{image_suffix}_w_refine.{image_ending}' if not save_name else save_name
            plt.savefig(os.path.join(save_folder, individual_save_name), bbox_inches='tight', pad_inches=0, dpi=300)
        elif output_mode == 'gif':
            # Append figure for GIF
            fig.canvas.draw()
            image_from_plot = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            images_for_gif.append(image_from_plot)

        plt.close(fig)

    if output_mode == 'gif':
        # Save all collected images as a GIF
        gif_save_name = f'{datetime_list[0]}_{datetime_list[-1]}_{image_suffix}_animation.gif' if not save_name else save_name
        images_for_gif[0].save(os.path.join(save_folder, gif_save_name), save_all=True, append_images=images_for_gif[1:], optimize=False, duration=500, loop=0)


def get_colors_from_colormap(colormap_name, num_colors):
    """
    Generates a list of color strings from a specified colormap in matplotlib.
    
    Parameters:
        colormap_name (str): Name of the colormap (e.g., 'jet', 'viridis', etc.).
        num_colors (int): Number of colors to generate.
    
    Returns:
        list: A list of color strings in hexadecimal format.
    """
    # Load the colormap
    cmap = plt.get_cmap(colormap_name)
    
    # Generate a list of points along the color map range
    colors = cmap([i / (num_colors - 1) for i in range(num_colors)])
    
    # Convert RGBA colors to hexadecimal format
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]
    
    return hex_colors



def create_track_res(selected_rip_boxes, image_folder, output_folder='output', output_filename='', suffix='PW', ending='jpg',
                         box_color='r', box_thickness=2, label = 'Rip', print_label = False, output_mode = 'gif'):
    """
    Create a GIF from photos with annotated timestamps and drawn boxes.

    Args:
    timestamps (list of str): List of timestamp strings in the format 'YYYY-MM-DD-HH-MM-SS'.
    boxes (list of list of tuples): List of boxes corresponding to each timestamp.
                                    Each box is a tuple (x1, y1, x2, y2).
    folder (str): Folder where photos are saved.
    output_folder (str): Folder where the output GIF will be saved.
    output_filename (str): Filename for the output GIF.
    suffix (str): Suffix of the photo filenames. Default is 'PW'.
    ending (str): File extension of the photo filenames. Default is 'jpg'.
    """
    os.makedirs(output_folder, exist_ok=True)
    images_for_gif = []  # List to store images if saving as GIF
    
    frame_num = selected_rip_boxes.get_frame_num()
    box_series = selected_rip_boxes.get_boxes()
    for i in range(frame_num):
        temp_box_object = box_series[i]
        box = temp_box_object.get_box()
        timestamp = temp_box_object.get_timestamp()
        temp_image = Image.open(os.path.join(image_folder, f"{timestamp}-{suffix}.{ending}"))
        temp_pval = temp_box_object.get_pval()

        image_width, image_height = temp_image.size
        # Plotting
        fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=300)
        ax.imshow(temp_image)
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=box_thickness, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        if print_label:
            label_text = f"{label}: {temp_pval:.3f}"
            ax.text(x1, y2, label_text, fontsize=12, verticalalignment='top', color='white', bbox=dict(facecolor='red', alpha=0.5))
        
        ax.text(0.5, 0.08, timestamp, transform=ax.transAxes, fontsize=24,
            horizontalalignment='center', verticalalignment='bottom',
            color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        if output_mode == 'images':
            # Save each figure individually
            individual_save_name = f'{timestamp}-{suffix}_track_boxes.{ending}' if not output_filename else output_filename
            plt.savefig(os.path.join(output_folder, individual_save_name), bbox_inches='tight', pad_inches=0, dpi=300)
        elif output_mode == 'gif':
            # Append figure for GIF
            fig.canvas.draw()
            image_from_plot = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            images_for_gif.append(image_from_plot)

        plt.close(fig)

    if output_mode == 'gif':
        # Save all collected images as a GIF
        gif_save_name = f'{selected_rip_boxes.get_start_time()}_{selected_rip_boxes.get_end_time()}_{suffix}_track_boxes.gif' if not output_filename else output_filename
        images_for_gif[0].save(os.path.join(output_folder, gif_save_name), save_all=True, append_images=images_for_gif[1:], optimize=False, duration=500, loop=0)
    