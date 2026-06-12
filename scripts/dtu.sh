dataset_folder=/home/wangsc/Documents/datasets/dtu_dataset/dtu
output_folder=/media/data/SurR/outputs/gausr/dtu_wo-mvc
for scene in 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
do
    python train.py -s ${dataset_folder}/scan${scene} -m ${output_folder}/scan${scene} -r 2 --use_decoupled_appearance 3
    python mesh_extract.py -m ${output_folder}/scan${scene}
    python evaluate_dtu_mesh.py -m ${output_folder}/scan${scene}
done