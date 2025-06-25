import os
from PIL import Image
import matplotlib.pyplot as plt


base_dir = 'C:\\Users\\yahaf\\OneDrive\\Desktop\\comparison'
gt_dir = os.path.join(base_dir, 'Ground Truth')
clip_dir = os.path.join(base_dir, 'ViT')


gt_images = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])


num_images = len(gt_images)
num_subjects = 5


fig, axes = plt.subplots(num_images, num_subjects + 1, figsize=(3*(num_subjects+1), 3*num_images))


column_titles = ['Ground Truth'] + [f'Subj {i}' for i in range(1, num_subjects + 1)]
for ax, title in zip(axes[0], column_titles):
    ax.set_title(title, fontsize=14)


for row_idx, gt_filename in enumerate(gt_images):
    image_id = gt_filename.replace('_ground_truth.png', '')
    
    
    gt_path = os.path.join(gt_dir, gt_filename)
    gt_img = Image.open(gt_path)
    axes[row_idx][0].imshow(gt_img)
    axes[row_idx][0].axis('off')
    
    
    for subj in range(1, num_subjects + 1):
        gen_path = os.path.join(clip_dir, str(subj), f'{image_id}_generated_vit.png')
        if os.path.exists(gen_path):
            gen_img = Image.open(gen_path)
            axes[row_idx][subj].imshow(gen_img)
        else:
            axes[row_idx][subj].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
        axes[row_idx][subj].axis('off')


plt.tight_layout()
plt.savefig("vit_vs_gt_qualitative_results.png", dpi=300)
plt.show()
