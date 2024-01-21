target_domain=${1}

python3 main.py \
--experiment=Domain_Adaptation \
--experiment_name=Domain_Adaptation/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=64 \
--num_workers=5 \
--grad_accum_steps=1
