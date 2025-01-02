if __name__ == '__main__':
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict 
    # do the training and checkpoint saving
    import sys
    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2]
    print(f"get state dict from {checkpoint_dir} and save to {output_dir}")
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir,output_dir,max_shard_size="200GB",exclude_frozen_parameters=True)
    print("Done")