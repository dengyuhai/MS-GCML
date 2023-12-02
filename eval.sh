python ./code/eval.py --save_feat 1 --dataset_name ovor --resume_name clip_ft.pth --gpu_id 0 
python ./code/eval.py --save_feat 0 --dataset_name ovor --resume_name clip_ft.pth --gpu_id 1 
python ./code/eval.py --save_feat 1 --dataset_name lvis --resume_name clip_ft.pth --gpu_id 0 
python ./code/eval.py --save_feat 0 --dataset_name lvis --resume_name clip_ft.pth --gpu_id 1 
python ./code/eval.py --save_feat 1 --dataset_name VG --resume_name clip_ft.pth --gpu_id 0 