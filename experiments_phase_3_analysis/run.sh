python -u compare_two_cl_runs_all_steps.py \
  --dataset VGGSound_hard2easy_balance \
  --modality audio-visual \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data_hard2easy/balance \
  --num_classes 100 \
  --class_num_per_step 10 \
  --model_a_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_wo_lowerbound/save/VGGSound_hard2easy_balance_lowerbound_c_wo_seed42 \
  --model_b_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_z2contrastive/save/VGGSound_hard2easy_balance_c_z2_contrastive_seed42 \
  --model_a_name w_o \
  --model_b_name z2_alpha2 \
  --out_root analysis/h2e_wo_vs_z2_all_steps \
  --infer_batch_size 64 \
  --num_workers 1 \
  --topk_centroid 5 \
  --knn_k 10 \
  --cache_feature_steps 2,5,9

  python -u compare_two_cl_runs_all_steps.py \
  --dataset VGGSound_easy2hard_balance \
  --modality audio-visual \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data_easy2hard/balance \
  --num_classes 100 \
  --class_num_per_step 10 \
  --model_a_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_wo_lowerbound/save/VGGSound_easy2hard_balance_lowerbound_c_wo_seed42 \
  --model_b_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_z2contrastive/save/VGGSound_easy2hard_balance_c_z2_contrastive_seed42 \
  --model_a_name w_o \
  --model_b_name z2_alpha2 \
  --out_root analysis/e2h_wo_vs_z2_all_steps \
  --infer_batch_size 64 \
  --num_workers 1 \
  --topk_centroid 5 \
  --knn_k 10 \
  --cache_feature_steps 2,5,9


  python -u compare_two_cl_runs_all_steps.py \
  --dataset VGGSound_balance_random1 \
  --modality audio-visual \
  --feature_root ../../../datasets/VGGSound \
  --meta_root ../data_balance_random/random1 \
  --num_classes 100 \
  --class_num_per_step 10 \
  --model_a_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_wo_lowerbound/save/VGGSound_data2_balance_lowerbound_c_wo_seed42 \
  --model_b_ckpt_dir /scratch/ganymede2/dal400642/Projects/AV-CIL_ICCV2023/experiments_phase_3_z2contrastive/save/VGGSound_data2_balance_c_z2_contrastive_seed42 \
  --model_a_name w_o \
  --model_b_name z2_alpha2 \
  --out_root analysis/random1_wo_vs_z2_all_steps \
  --infer_batch_size 64 \
  --num_workers 1 \
  --topk_centroid 5 \
  --knn_k 10 \
  --cache_feature_steps 2,5,9