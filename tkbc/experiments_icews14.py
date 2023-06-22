#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'d'}])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python3 ../learner.py '\
        f'--dataset ICEWS14 '\
        f'--model TNTComplEx '\
        f'--rank {c["rank"]} --emb_reg {c["emb_reg"]} --time_reg {c["time_reg"]} --time_norm {c["time_norm"]} --time_reg_w {c["time_reg_w"]} --p_norm {c["p_norm"]}'
    return command


def to_logfile(c, path):
    outfile = "{}/icews14.{}.log".format(path, summary(c).replace("/", "_")) #change here for other datasets
    return outfile


def main(argv):
    hyp_space = [dict(
        rank=[5, 25, 50, 100, 500, 2000],
        emb_reg=[1e-1, 1e-2, 1e-3, 1e-4],
        time_reg_w=[1, 1e-1, 1e-2, 1e-3, 1e-4],
        time_norm=['Lp'],
        p_norm=[1, 2, 3, 4, 5],
        time_reg=['smooth']
    ),
        dict(
            rank=[5, 25, 50, 100, 500, 2000],
            emb_reg=[1e-1, 1e-2, 1e-3, 1e-4],
            time_reg_w=[1, 1e-1, 1e-2, 1e-3, 1e-4],
            time_norm=['Np'],
            p_norm=[1, 2, 3, 4, 5],
            time_reg=['smooth']
        )]

    configurations = list(cartesian_product(hyp_space[argv[1]]))

    path = 'logs/icews14'

    # If the folder that will contain logs does not exist, create it
    #if not os.path.exists(path):
        #os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = ('Training finished' in content) or ('Loss is nan: nan' in content)

        if not completed:
            cmd = to_cmd(cfg)
            if cmd is not None:
                command_line = f'{cmd} > {logfile} 2>&1'
                command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = f"""#!/usr/bin/env bash

#SBATCH --output=/home/%u/slogs/tntcomplex-%A_%a.out
#SBATCH --error=/home/%u/slogs/tntcomplex-%A_%a.err
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=14GB # memory
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH -t 6:00:00 # time requested in hours:minutes:seconds
#SBATCH --array 1-{nb_jobs}

echo "Setting up bash environment"
source ~/.bashrc
set -e # fail fast

conda activate mypt

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/projects/tntcomplex/tkbc/scripts

"""

    if header is not None:
        print(header)

    is_slurm = True
    for job_id, command_line in enumerate(sorted_command_lines, 1):
        if is_slurm:
            print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}')
        else:
            print(f'{command_line}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])