##!/bin/bash
#
## Check if the user provided an argument
#if [ "$#" -ne 2 ]; then
#    echo "Usage: $0 <output_folder> <config_file> "
#    exit 1
#fi
#
#
#output_folder="$1"
#config_file="$2"
#
#if [ ! -d "$output_folder" ]; then
#    echo "Error: Folder '$output_folder' does not exist."
#    exit 2
#fi
#
#
#
## Remove the selected object
#python edit_object_removal.py -m ${output_folder} --config_file ${config_file}

#!/bin/bash

# 1. 校验必要参数：至少需要 2 个（output_folder 和 config_file），--skip_test 是可选参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <output_folder> <config_file> [--skip_test]"  # 提示可选参数
    exit 1
fi

# 2. 提取必要参数（前 2 个参数固定为输出目录和配置文件）
output_folder="$1"
config_file="$2"

# 3. 检查可选参数：是否包含 --skip_test
skip_test=""
if [ "$#" -eq 3 ] && [ "$3" = "--skip_test" ]; then
    skip_test="--skip_test"  # 捕获可选参数，后续传给 Python 脚本
fi

# 4. 校验输出目录是否存在（原有逻辑保留）
if [ ! -d "$output_folder" ]; then
    echo "Error: Folder '$output_folder' does not exist."
    exit 2
fi

# 5. 执行 Python 脚本：传递必要参数 + 可选的 --skip_test
python edit_object_removal0.py -m ${output_folder} --config_file ${config_file} ${skip_test}
#!!!记得加--skip_test
