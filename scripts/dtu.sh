dataset_folder=/home/wangsc/Documents/datasets/dtu_dataset/dtu
for scene in 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
do
#    python train.py -s ${dataset_folder}/scan${scene} -m /media/data/SurR/outputs/gausr/dtu/scan${scene} -r 2 --use_decoupled_appearance 3
    python mesh_extract.py -m /media/data/SurR/outputs/gausr/dtu/scan${scene}
    python evaluate_dtu_mesh.py -m /media/data/SurR/outputs/gausr/dtu/scan${scene}
done