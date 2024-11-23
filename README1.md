# Holistic CLIP "Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training"
## Get Started
Our code is adapted from [open_clip](https://github.com/mlfoundations/open_clip), please refer to this repository for environment setup. For synthetic data generation, please follow the environment setup of [LLaVA1.5](https://github.com/haotian-liu/LLaVA), [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4), [Internvl2](https://huggingface.co/OpenGVLab/InternVL2-2B) and [QwenVL2](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

## Synthetic Data Generation
In **synthetic_caption**, we provide the code and instruction for generating image captions using different VLMs: LLaVA1.5, MiniGPT4, Internvl2 and QwenVL2. For multi-prompt generation, we use Internvl2 as MLLM backbone and the prompts are already included in the file. We allow multi-gpus inference without the need for manual split. More details for synthetic data preparation is in **synthetic_caption/README.md**.

## Model Training
In **src_cls** / **src_mlp**, we provide the training/inference code for cls/mlp based multi-to-multi contrastive learning models. The training jsonlines file should be in the form of {"image":img_path,"caption":\[caption1,caption2,...,captionM\]}. 
In normal case, set **cls_num** / **moe_head** to M (if not equal to M, the matching will not be one-to-one, but free-matching.)

## Model Inference