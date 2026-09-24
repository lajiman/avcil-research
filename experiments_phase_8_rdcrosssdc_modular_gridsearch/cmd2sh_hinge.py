import os

import subprocess

with open('grid_commands/commands_hinge.txt', 'r') as direct_f:
    lines = direct_f.readlines()

    n = len(lines)
    n_per_gpu = n // 4
    res = n - n_per_gpu * 4

    lines_0 = list(range(n_per_gpu*0, n_per_gpu*1))
    lines_1 = list(range(n_per_gpu*1, n_per_gpu*2))
    lines_2 = list(range(n_per_gpu*2, n_per_gpu*3))
    lines_3 = list(range(n_per_gpu*3, n_per_gpu*4))

    lines_0.append(80)

    print(lines_0, lines_1, lines_2, lines_3)

    sh_0 = 'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n'
    sh_1 = 'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n'
    sh_2 = 'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n'
    sh_3 = 'export OPENBLAS_NUM_THREADS=1\nexport OMP_NUM_THREADS=8\nexport MKL_NUM_THREADS=8\n'

    for i in range(n):
        if i in lines_0:
            gpu_id = 0
            idx = lines_0.index(i)
            sub_line_len = len(lines_0)
        elif i in lines_1:
            gpu_id = 1
            idx = lines_1.index(i)
            sub_line_len = len(lines_1)
        elif i in lines_2:
            gpu_id = 2
            idx = lines_2.index(i)
            sub_line_len = len(lines_2)
        elif i in lines_3:
            gpu_id = 3
            idx = lines_3.index(i)
            sub_line_len = len(lines_3)
        else:
            raise ValueError('{}'.format(i))
        
        cmd = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_id) + lines[i].strip()

        cmd += ' &'
        cmd += '\n'

        if idx % 2 != 0 or idx == sub_line_len-1:
            cmd += '\nwait\nsleep 3\n\n'

        if i in lines_0:
            sh_0 += cmd
        elif i in lines_1:
            sh_1 += cmd
        elif i in lines_2:
            sh_2 += cmd
        elif i in lines_3:
            sh_3 += cmd
        else:
            raise ValueError('{}'.format(i))

    sh_0 += 'echo \'GPU0 jobs done!!!!!!!!!!!!!!!!!\''
    sh_1 += 'echo \'GPU1 jobs done!!!!!!!!!!!!!!!!!\''
    sh_2 += 'echo \'GPU2 jobs done!!!!!!!!!!!!!!!!!\''
    sh_3 += 'echo \'GPU3 jobs done!!!!!!!!!!!!!!!!!\''

with open('./run_hinge_0.sh', 'w') as f:
    f.write(sh_0)

with open('./run_hinge_1.sh', 'w') as f:
    f.write(sh_1)

with open('./run_hinge_2.sh', 'w') as f:
    f.write(sh_2)

with open('./run_hinge_3.sh', 'w') as f:
    f.write(sh_3)

    



