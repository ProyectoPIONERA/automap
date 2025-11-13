#!/bin/bash

project="/home/carlos/workspace/automap"
cache_dir="$HOME/.cache/automap"

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

# Conda env
if [ -z "$CONDA_PREFIX" ]; then
    source /home/carlos/miniconda3/bin/activate $project/.venv
fi

# Directories paths
automap="$project/automap"
dataset="$project/datasets/PODIO"
# -------------------------
scenarios="discursos_navidad"
# scenarios="scenario1/1A scenario1/1B"

# For a single run: use '_' or '-' or other char different from '/' (no slash to avoid creating subdirectories).
# For multiple runs: use '/run1 /run2 /run3'. The slash is important to create different subdirectories.
runs="_"
# -------------------------

# -------------------------
# EXECUTABLES
# Python
python="python"

# Method
method="$python $automap/methods/example/chatgpt.py"

# Evaluation
compute_metrics="$python $automap/grapheval/compute_metrics.py"

# Postprocessing
map2rml="$python $automap/converters/map2rml.py"
rml_mapper="java -jar $project/resources/rmlmapper-8.0.0-r378-all.jar"

#Visualization
eval2tabular="$python $automap/utils/eval2tabular.py"
eval2wb="$python $automap/utils/eval2wandb.py"
# -------------------------

echo =========================================================================
echo "Starting experiments $(date +"%Y-%m-%d %H:%M:%S")"
echo =========================================================================

for scenario in $scenarios; do
    for run in $runs; do
        data="$dataset/data/$scenario"
        exp="chatgpt$run"
        exp_dir="$dataset/exps/$scenario/$exp"
        echo "========================================================================"
        echo "Processing scenario: $scenario | Run: $run | Method: $method $(date +"%Y-%m-%d %H:%M:%S")"
        echo "------------------------------------------------------------------------"
        
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
        input_files="$data/discursos_navidad.csv"

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

        # $remap \
        #     --csv $input_files \
        #     --rdf $gold_graph_path \
        #     --output $mapping_rml_path 2> $logs_path/remap.log

        $method \
            --csv $input_files \
            --output $mapping_rml_path 2> $logs_path/$method.log

        # ==============================================================================
        # ==============================================================================
        # MAPPING TO RML AND GRAPH GENERATION
        # ==============================================================================

        # ReMap already outputs RML
        # echo =========================================================================
        # echo "Converting mapping from YAML to RML $(date +"%Y-%m-%d %H:%M:%S")"
        # echo =========================================================================
        # cat $mapping_yml_path | $map2rml > $mapping_rml_path 2> $logs_path/map2rml.log

        echo "------------------------------------------------------------------------"
        echo "Creating Knowledge Graph $(date +"%Y-%m-%d %H:%M:%S")"
        echo "------------------------------------------------------------------------"
        touch $pred_graph_path
        $rml_mapper \
            -m $mapping_rml_path \
            -o $pred_graph_path 2> $logs_path/rmlmapper.log

        # ==============================================================================

        # ==============================================================================
        # EVALUATION
        # ==============================================================================

        echo "------------------------------------------------------------------------"
        echo "Starting evaluation $(date +"%Y-%m-%d %H:%M:%S")"
        echo "------------------------------------------------------------------------"

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
echo =========================================================================
echo "Finished experiments $(date +"%Y-%m-%d %H:%M:%S")"
echo =========================================================================
