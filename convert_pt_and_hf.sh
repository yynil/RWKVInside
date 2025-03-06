#!/bin/bash

# 定义默认值
model_path="/home/yueyulin/models/DeepSeek-R1-Distill-Qwen-7B/"
ckpt_file=""
ref_dir="/home/yueyulin/model/qwen_r1_7b_gatefree_freezemlp_hf/"
output_dir=""
copy_mlp_from_original=""
config_file=""
wkv_has_norm=""
wkv_version="7"
wkv_has_gate=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      model_path="$2"
      shift 2
      ;;
    --ckpt_file)
      ckpt_file="$2"
      shift 2
      ;;
    --ref_dir)
      ref_dir="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --copy_mlp_from_original)
        copy_mlp_from_original="--copy_mlp_from_original"
        shift
        ;;
    --config_file)
        config_file="$2"
        shift 2
        ;;
    --wkv_has_norm)
        wkv_has_norm="--wkv_has_norm"
        shift
        ;;
    --wkv_version)
        wkv_version="$2"
        shift 2
        ;;
    --wkv_has_gate)
        wkv_has_gate="--wkv_has_gate"
        shift
        ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查必要参数
if [ -z "$ckpt_file" ]; then
  echo "错误: 必须指定 --ckpt_file 参数"
  exit 1
fi

if [ -z "$output_dir" ]; then
  echo "错误: 必须指定 --output_dir 参数"
  exit 1
fi

if [ -z "$config_file" ]; then
  echo "错误: 必须指定 --config_file 参数"
  exit 1
fi

# 检查ckpt_file是否存在
if [ ! -f "$ckpt_file" ]; then
  echo "错误: 指定的检查点文件 '$ckpt_file' 不存在"
  exit 1
fi

# 如果output_dir存在则删除，然后创建新的
if [ -d "$output_dir" ]; then
  echo "删除现有目录: $output_dir"
  rm -rf "$output_dir"
fi

echo "创建输出目录: $output_dir"
mkdir -p "$output_dir"

echo "参数检查通过，开始处理..."
echo "模型路径: $model_path"
echo "检查点文件: $ckpt_file"
echo "参考目录: $ref_dir"
echo "输出目录: $output_dir"

# 这里可以添加转换模型的实际代码

# 转化模型第一步，去掉 teacher
#创建临时目录$output_dir_tmp
output_dir_tmp="${output_dir}_tmp"
#删除现有目录$output_dir_tmp
echo "删除现有目录: $output_dir_tmp"
rm -rf "$output_dir_tmp"
#创建临时目录$output_dir_tmp
echo "创建临时目录: $output_dir_tmp"
mkdir -p "$output_dir_tmp"
echo "转化模型第一步，去掉 teacher"
python train_scripts/convert_pt.py --original_model_path $model_path\
  --model_path "$ckpt_file" \
  --output_path "$output_dir_tmp" $copy_mlp_from_original

# 从 ref_dir 拷贝文件到 output_dir
echo "从 $ref_dir 拷贝文件到 $output_dir"
cp -r "$ref_dir"/*json "$output_dir"

# 转化成hf格式
echo "转化成hf格式, wkv_version: $wkv_version, wkv_has_gate: $wkv_has_gate, wkv_has_norm: $wkv_has_norm, config_file: $config_file, output_dir: $output_dir,ckpt_file: $output_dir_tmp "
python test/convert_2_hf.py \
    --config_file "$config_file" \
    --ckpt_file "$output_dir_tmp" \
    $wkv_has_norm \
    --wkv_version "$wkv_version" \
    $wkv_has_gate \
    --output_config_dir "$output_dir"

# 删除临时目录$output_dir_tmp
echo "删除临时目录: $output_dir_tmp"
rm -rf "$output_dir_tmp"