dataset_folder=/home/wangsc/Documents/datasets/translab/
output_folder=/media/data/SurR/outputs/gausr/trans-mvg-aspt
eval_folder=/home/wangsc/Documents/datasets/translab/
for scene in 05 06 07 08
do
    # python train.py -s ${dataset_folder}/scene_${scene} -m ${output_folder}/scene_${scene} -r 2
    # python mesh_extract.py -m ${output_folder}/scene_${scene}
    python scripts/eval_translab/eval.py --data ${output_folder}/scene_${scene}/recon_post.ply --scan scene_${scene} --vis_out_dir ${output_folder}/scene_${scene} --dataset_dir ${dataset_folder}/scene_${scene} --mode mesh --downsample_density 0.002
done