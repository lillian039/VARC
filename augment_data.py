from utils.data_augmentation import augment_raw_data_split_per_task

if __name__ == "__main__":
    augment_raw_data_split_per_task(
        dataset_root="raw_data/ARC-AGI",
        split="evaluation",
        output_subdir="eval_color_permute_ttt_9",
        num_permuate=9,
        only_basic=True,
    )

    augment_raw_data_split_per_task(
        dataset_root="raw_data/ARC-AGI-2",
        split="evaluation",
        output_subdir="eval_color_permute_ttt_9",
        num_permuate=9,
        only_basic=True,
    )