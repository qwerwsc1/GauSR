ulimit -n 4096
dataset_folder=/home/wangsc/Documents/datasets/tnt_dataset/
output_folder=/media/data/SurR/outputs/gausr/tnt-wo-assumption
scenes=(Barn Caterpillar Ignatius Meetingroom Truck Courthouse)
devices=(cuda cuda cuda cuda cuda cuda)

for idx in "${!scenes[@]}"; do
    scene="${scenes[$idx]}"
    device="${devices[$idx]}"
    python train.py -s ${dataset_folder}/${scene} -m ${output_folder}/${scene} -r 2 --use_decoupled_appearance 3 --data_device ${device}
    python mesh_extract_tnt.py -m ${output_folder}/${scene} --use_depth_filter
    python eval_tnt/run.py --dataset-dir ${dataset_folder}/${scene} --traj-path ${dataset_folder}/${scene}/${scene}_COLMAP_SfM.log --ply-path ${output_folder}/${scene}/recon_post.ply --out-dir ${output_folder}/${scene}/mesh
done