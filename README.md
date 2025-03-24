<p align="center" width="100%">
<img src="docs/logo.png" alt="Example Image" width="25" style="margin-bottom:-8px"/> MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios that are Difficult for Humans?
</p>



<font size=7><div align='center' >  
[[📖 arXiv Paper](https://arxiv.org/abs/2502.10391)] 
[[📊 MM-Unify Data](https://huggingface.co/datasets/wulin222/MME-Unify)] 
[[📝 Homepage](https://aba122.github.io/MME-Unify.github.io/)] 

</div></font>

# A Comprehensive Benchmark for Unified Multimodal Understanding and Generation


**[2025/02/10]** 🔥 We are proud to open-source **MM-Unify**, a comprehensive evaluation framework designed to systematically assess U-MLLMs. Our benchmark includes:

- A **Standardized Traditional Task Evaluation** We sample from 12 datasets, covering 10 tasks with 30 subtasks, ensuring consistent and fair comparisons across studies.
- A **Unified Task Assessment** We introduce five novel tasks testing multimodal reasoning, including image editing, commonsense QA with image generation, and geometric reasoning.
- A **Comprehensive Model Benchmarking** We evaluate 12 leading U-MLLMs, such as Janus-Pro, EMU3, and VILA-U, alongside specialized understanding (e.g., Claude-3.5) and generation models (e.g., DALL-E-3).

Our findings reveal substantial performance gaps in existing U-MLLMs, highlighting the need for more robust models capable of handling mixed-modality tasks effectively.


<p align="center">
    <img src="docs/Bin.png" width="100%" height="100%">
</p>

## Dataset Examples

<p align="center">
    <img src="docs/main.png" width="100%" height="100%">
</p>


## Evaluation Pipeline

### Prompt ###
The common prompt used in our evaluation for different tasks can be found in:
```bash
MME-Unify/Prompt.txt
```

### Dataset ###
You can download images in our [Hugging Face repository](https://huggingface.co/datasets/wulin222/MME-Unify) and the final structure should look like this:

```
MME-Unify
├── CommonSense_Questions
├── Conditional_Image_to_Video_Generation
├── Fine-Grained_Image_Reconstruction
├── Math_Reasoning
├── Multiple_Images_and_Text_Interlaced
├── Single_Image_Perception_and_Understanding
├── Spot_Diff
├── Text-Image_Editing
├── Text-Image_Generation
├── Text-to-Video_Generation
├── Video_Perception_and_Understanding
└── Visual_CoT

```

You can found QA pairs in:
```bash
MME-Unify/Unify_Dataset
```
and the structure should look like this:
```
Unify_Dataset
├── Understanding
├── Generation
├── Unify_Capability
│   ├── Auxiliary_Lines
│   ├── Common_Sense_Question
│   ├── Image_Editing_and_Explaning
│   ├── SpotDiff
│   ├── Visual_CoT

```

### Evaluate ###
To extract the answer and calculate the scores, we add the model response to a JSON file. Here we provide an example template [output_test_template.json](./evaluation/output_test_template.json). Once you have prepared the model responses in this format, please refer to the evaluation scripts in:
```bash
MME-Unify/Evaluate
```

## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex
@article{zhang2025mm,
  title={MM-RLHF: The Next Step Forward in Multimodal LLM Alignment},
  author={Zhang, Yi-Fan and Yu, Tao and Tian, Haochen and Fu, Chaoyou and Li, Peiyan and Zeng, Jianshu and Xie, Wulin and Shi, Yang and Zhang, Huanyu and Wu, Junkang and others},
  journal={arXiv preprint arXiv:2502.10391},
  year={2025}
}
```
