# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/eval_policy.sh dp3 rlbench_open_door 0322 0 2 joint_positions gripper_states pcd_from_mesh
# bash scripts/eval_policy.sh dp3 rlbench_open_door 0322 1 2  gripper_states joint_positions front_pcd


DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"
gpu_id=${5}
state=${6}
action=${7}
pcd=${8}

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            state=${state} \
                            action=${action} \
                            pcd=${pcd}



                                