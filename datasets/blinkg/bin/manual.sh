#!/bin/zsh

project="$HOME/workspace/automap"
cache_dir="$HOME/.cache/automap"

# Function to print formatted messages with timestamp
log_message() {
    local msg="$1"
    echo "------------------------------------------------------------------------"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")]: $msg"
    echo "------------------------------------------------------------------------"
}

# Parse command line arguments
no_backup=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-backup)
            no_backup=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-backup]"
            exit 1
            ;;
    esac
done

# SET TRUE ONLY FOR DEBUGGING PURPOSES
no_backup=true

# Conda env
eval "$(conda shell.bash hook)"
conda activate $project/.venv

# Directories paths
automap="$project/automap"
dataset="$project/datasets/blinkg"
# -------------------------
scenarios="scenario1/1B"

# For a single run: use '_' or '-' or other char different from '/' (no slash to avoid creating subdirectories).
# For multiple runs: use '/run1 /run2 /run3'. The slash is important to create different subdirectories.
runs="_"
# -------------------------

# -------------------------
# EXECUTABLES
# Python
python="python"

# Method
method="$python $automap/methods/examples/manual.py"
method_name="manual"

# Evaluation
compute_metrics="$python $automap/grapheval/compute_metrics.py"

# Postprocessing
map2rml="$python $automap/converters/map2rml.py"
rml_mapper="java -jar $project/resources/rmlmapper-8.0.0-r378-all.jar"

#Visualization
eval2tabular="$python $automap/utils/eval2tabular.py"
eval2wb="$python $automap/utils/eval2wandb.py"
# -------------------------

log_message "Starting experiments"

for scenario in $scenarios; do
    for run in $runs; do
        data="$dataset/data/$scenario"
        exp="${method_name}$run"
        exp_dir="$dataset/exps/$scenario/$exp"
        log_message "Processing scenario: $scenario | Run: $run"
        
        # Backup and reset exp dir if exists
        if [ "$no_backup" = false ]; then
            mkdir -p "$cache_dir/exp_backups"
            echo "Backing up existing experiment to $cache_dir/exp_backups/${exp}_backup_$(date +"%Y%m%d_%H%M%S")"
            mv -f "$exp_dir" "$cache_dir/exp_backups/${exp}_backup_$(date +"%Y%m%d_%H%M%S")"
        else
            echo "No backup requested. Overwriting existing experiment directory."
            rm -rf "$exp_dir"
        fi

        # Create experiment directory
        mkdir -p "$exp_dir"
        mkdir -p "$exp_dir/data"

        # Link data to exp directory. Just for easier access when reviewing results by hand.
        ln -sf $data/* $exp_dir/data
        ln -sf $data/../config.yaml $exp_dir/data/config.yaml

        # ==============================================================================
        # Data paths
        input_files="$data/*.csv"

        # Results paths
        mapping_yml_path="$exp_dir/mapping.yml"
        mapping_rml_path="$exp_dir/mapping.rml.ttl"
        pred_graph_path="$exp_dir/graph.nt"
        gold_graph_path="$data/gold_graph.nt"
        eval_results_path="$exp_dir/eval_results.json"
        tabular_results_path="$exp_dir/eval_results.tsv"

        logs_path="$exp_dir/logs"
        mkdir -p $logs_path
        # ==============================================================================

        # ==============================================================================
        # PREDICTIONS
        # ==============================================================================
        log_message "Executing method"

        $method > $mapping_rml_path 2> $logs_path/$method_name.log

        # ==============================================================================
        # ==============================================================================
        # MAPPING TO RML AND GRAPH GENERATION
        # ==============================================================================

        # ReMap already outputs RML
        # log_message "Converting mapping from YAML to RML"
        # cat $mapping_yml_path | $map2rml > $mapping_rml_path 2> $logs_path/map2rml.log

        log_message "Creating Knowledge Graph"
        touch $pred_graph_path
        $rml_mapper \
            -m $mapping_rml_path \
            -o $pred_graph_path 2> $logs_path/rmlmapper.log

        # ==============================================================================

        # ==============================================================================
        # EVALUATION
        # ==============================================================================

        log_message "Starting evaluation"

        # Temp. It is needed to sort the files. This should be fixed in the code.
        cat $gold_graph_path | sort > $exp_dir/gold_graph.nt.sorted
        mv $exp_dir/gold_graph.nt.sorted $exp_dir/gold_graph.nt

        cat $pred_graph_path | sort > $pred_graph_path.sorted
        mv $pred_graph_path.sorted $pred_graph_path

        cat $pred_graph_path | $compute_metrics \
            --config $data/../config.yaml \
            --pred_mapping $mapping_rml_path \
            --gold_graph $gold_graph_path > $eval_results_path
        # ==============================================================================

        # ==============================================================================
        # VISUALIZATION
        # ==============================================================================
        cat $eval_results_path | $eval2tabular > $tabular_results_path
    done
done
log_message "Finished experiments"
