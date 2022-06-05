#!/bin/bash

# # bc
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_bc_official_test1/model.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_bc_test1_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # 10%bc
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_pbc_official_test1/model_196607.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_pbc_test1_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # 20%bc
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_pbc_official_test2/model_196607.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_pbc_test2_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # 30%bc
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_pbc_official_test3/model_393215.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_pbc_test3_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # IQL_tau=0.5
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test1/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test1_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test1/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test1_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test1/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test1_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# === BREAK ===

# # IQL_tau=0.7
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# # done.
python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
                      model.seperate_policy=true model.seperate_target=true \
                      evaluator.generation_kwargs.include_adv=true \
                      evaluator.generation_kwargs.adv_weight=16.0 \
                      eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_beta16_beam1_eval1_4.pkl \
                      evaluator.env.url=http://localhost:5002/step_rank

# # IQL_tau=0.8
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test3/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test3_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test3/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test3_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test3/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test3_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# # IQL_tau=0.9
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test4/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test4_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test4/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test4_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test4/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test4_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # CQL_alpha=0.1
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test1/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test1_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test1/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test1_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test1/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test1_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test1/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test1_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # CQL_alpha=1.0
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test2_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# # === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test2_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test2_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test2_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # CQL_alpha=10.0
# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test3_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test3_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test3_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_cql_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_cql_test3_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# # === BREAK ===

# # PSI_alpha=0.1

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test1/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test1_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test1/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test1_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test1/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test1_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test1/model_262143.pkl  \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test1_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # PSI_alpha=1.0

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test2/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test2_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test2/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test2_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test2/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test2_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test2/model_327679.pkl  \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test2_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # PSI_alpha=10.0

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test3_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test3_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test3_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_psi_official_test3/model_262143.pkl  \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_phi_test3_betainf_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.generation_kwargs.max_generation_len=10

# === BREAK ===

# # standard BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_bc_official_test1/model.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_bc_test1_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard 10% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_pbc_official_test1/model_622591.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_pbc_test1_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard 20% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_pbc_official_test2/model_131071.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_pbc_test2_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard 30% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_pbc_official_test3/model_131071.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_pbc_test3_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard IQL_tau=0.5

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test1/model_196607.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test1_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test1/model_196607.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test1_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test1/model_196607.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test1_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard IQL_tau=0.7

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test2/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test2_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test2/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test2_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test2/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test2_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard IQL_tau=0.8

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test3/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test3_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test3/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test3_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test3/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test3_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # standard IQL_tau=0.9

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test4/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test4_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test4/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test4_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test4/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_iql_test4_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# === BREAK ===

# # conservative BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_bc_official_test1/model.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_bc_test1_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative 10% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_yn_pbc_official_test1/model_393215.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_pbc_test1_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative 20% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_yn_pbc_official_test2/model_491519.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_pbc_test2_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative 30% BC

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_yn_pbc_official_test3/model_262143.pkl \
#                       model.seperate_policy=false model.seperate_target=false \
#                       evaluator.generation_kwargs.include_adv=false \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_pbc_test3_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative IQL_tau=0.5

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test1/model_655359.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test1_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test1/model_655359.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test1_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test1/model_655359.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test1_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative IQL_tau=0.7

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test2/model_557055.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test2_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# === BREAK ===

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test2/model_557055.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test2_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test2/model_557055.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test2_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative IQL_tau=0.8

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test3/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test3_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test3/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test3_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test3/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test3_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # conservative IQL_tau=0.9

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test4/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test4_beta4_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test4/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test4_beta8_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test4/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_iql_test4_beta16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# === BREAK ===

# * Visdial hard y/n IQL_tau=0.7
# * Visual standard IQL_tau=0.7
# * Visual conservative IQL_tau=0.9

# IQL_tau=0.7 hard y/n to conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_to_conservative_iql_test2_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# # IQL_tau=0.7 hard y/n to standard

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_to_standard_iql_test2_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# === BREAK ===

# # IQL_tau=0.7 standard to hard y/n

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test2/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_to_hard_yn_iql_test2_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# # IQL_tau=0.7 standard to  conservative

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_standard_iql_official_test2/model_163839.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_standard_to_conservative_iql_test2_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=conservative \
#                       dataset.data.yn_reward_kind=conservative

# === BREAK ===

# # IQL_tau=0.9 conservative to hard y/n

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test4/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_to_hard_yn_iql_test4_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# # IQL_tau=0.9 conservative to standard

# # done.
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_conservative_iql_official_test4/model_491519.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_conservative_to_standard_iql_test4_beta16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=none \
#                       dataset.data.yn_reward_kind=none

# === BREAK ===

# # hard y/n AWAC_beta=4

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_awac_test1/model_294911.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_awac_test1_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# # hard y/n AWAC_beta=8

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_awac_test2/model_196607.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_awac_test2_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# # === BREAK ===

# hard y/n AWAC_beta=16

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_awac_test3/model_262143.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=false \
#                       evaluator.generation_kwargs.adv_weight=1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_awac_test3_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# # hard y/n utterance IQL_tau=0.5

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test1/model_1507327.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test1_gen4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test1/model_1507327.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test1_gen8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test1/model_1507327.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test1_gen16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# # hard y/n utterance IQL_tau=0.7

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test2_gen4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test2_gen8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test2/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test2_gen16_beam1_eval2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# # hard y/n utterance IQL_tau=0.8

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test3/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test3_gen4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test3/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test3_gen8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test3/model_229375.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test3_gen16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# hard y/n utterance IQL_tau=0.9

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test4/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test4_gen4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test4/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test4_gen8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_official_utterance_iql_test4/model_327679.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_utterance_test4_gen16_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# # IQL_tau=0.7, no_lm_base
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_no_lm_beta4_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_no_lm_beta8_beam1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_iql_official_test2/model_131071.pkl \
#                       model.seperate_policy=true model.seperate_target=true \
#                       evaluator.generation_kwargs.include_adv=true \
#                       evaluator.generation_kwargs.include_logits=false \
#                       evaluator.generation_kwargs.adv_weight=16.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_iql_test2_no_lm_beta16_beam1_eval1_2.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# === BREAK ===

# # DT
# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-11.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-11_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-10.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-10_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-9.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-9_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-8.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-8_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-7.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-7_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-6.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-6_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-5.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-5_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-4.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-4_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-3.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-3_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-2.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-2_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=-1.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r-1_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_dt_test1/model.pkl \
#                       evaluator.generation_kwargs.cond_r=0.0 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_dt_test1_r0_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank

# === BREAK ===

# # CHAI alpha=0.1

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test1_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test1_gen4_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test1_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test1_gen8_eval1.pkl \
#                       evaluator.env.url=http://localhost:5000/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test1_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test1_gen16_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# # CHAI alpha=1.0

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test2_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test2_gen4_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test2_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test2_gen8_eval1.pkl \
#                       evaluator.env.url=http://localhost:5001/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test2_3/model_458751.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test2_gen16_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# # CHAI alpha=10.0

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test3_3/model_491519.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=1 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test3_gen4_eval1.pkl \
#                       evaluator.env.url=http://localhost:5002/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# === BREAK ===

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test3_3/model_491519.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=2 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test3_gen8_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard

# python eval_policy.py model.load.checkpoint_path=outputs/visual_dialogue/visdial_hard_yn_chai_test3_3/model_491519.pkl \
#                       evaluator.generation_kwargs.num_generations=4 \
#                       evaluator.generation_kwargs.generation_batches=4 \
#                       eval.log_save_path=outputs/visual_dialogue/evals/visdial_hard_yn_chai_test3_gen16_eval1.pkl \
#                       evaluator.env.url=http://localhost:5003/step_rank \
#                       evaluator.env.yn_reward_kind=hard \
#                       dataset.data.yn_reward_kind=hard
