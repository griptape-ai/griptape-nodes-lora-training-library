# Griptape Nodes: LoRA Training Library

Train custom LoRA (Low-Rank Adaptation) models for FLUX.1 diffusion models within Griptape Nodes. Create personalized AI models by fine-tuning on your own images with automated dataset generation and AI-powered captioning.

## 🎯 Features

- **FLUX.1 Model Support**: Train LoRAs for FLUX.1-schnell, FLUX.1-dev, and FLUX.1-Krea-dev models
- **Automated Dataset Generation**: Convert images into properly structured training datasets
- **AI-Powered Captioning**: Automatically generate descriptive captions using GPT-4.1-mini
- **Manual Caption Support**: Option to provide your own custom captions
- **Advanced Training Parameters**: Full control over learning rates, epochs, network dimensions, and optimization settings
- **Memory Optimization**: Support for fp8 quantization, mixed precision training, and high VRAM mode
- **HuggingFace Integration**: Automatic model downloading and caching from HuggingFace Hub
- **Safetensors Format**: Modern, secure model format for saving trained LoRAs
- **Professional Training Pipeline**: Built on Kohya sd-scripts framework with Accelerate integration

## 📦 Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- Python 3.12 or higher
- CUDA-compatible GPU with sufficient VRAM (8GB+ recommended)
- Griptape Cloud API key (for AI captioning feature)

### Install the Library

1. **Download the library files** to your Griptape Nodes libraries directory:
   ```bash
   # Navigate to your Griptape Nodes libraries directory
   cd `gtn config show workspace_directory`
   
   # Clone or download this library
   git clone --recurse-submodules https://github.com/your-username/griptape-nodes-lora-training-library.git
   ```

2. **Add the library** in the Griptape Nodes Editor:
   * Open the Settings menu and navigate to the *Libraries* settings
   * Click on *+ Add Library* at the bottom of the settings panel
   * Enter the path to the library JSON file: **your Griptape Nodes Workspace directory**`/griptape-nodes-lora-training-library/griptape_nodes_lora_training_library/griptape-nodes-library.json`
     * Note: Select the `library.json` file based on your dependency preferences. For instance, `griptape-nodes-library-cuda129.json` defines dependencies for Cuda 12.9.
   * You can check your workspace directory with `gtn config show workspace_directory`
   * Close the Settings Panel
   * Click on *Refresh Libraries*

3. **Verify installation** by checking that the "Generate LoRA Dataset" and "Train LoRA" nodes appear in your Griptape Nodes interface in the "LoRA" category.

## 🔑 API Key Setup

### Griptape Cloud API Key (Required for AI Captioning)

If you want to use the automated captioning feature, you'll need a Griptape Cloud API key:

1. **Get your API key** from [Griptape Cloud](https://cloud.griptape.ai/)
2. **Configure the API key** in Griptape Nodes:
   * Open the *Settings* menu and navigate to *API Keys & Secrets*
   * Click on *+ Add Secret* to add a new secret
   * Set the key name as `GT_CLOUD_API_KEY`
   * Enter your API key value

Alternatively, you can set it as an environment variable:
```bash
export GT_CLOUD_API_KEY="your-api-key-here"
```

## 🚀 Usage

### Basic Workflow

The LoRA training process involves two main steps:

1. **Generate Dataset**: Convert your images into a training dataset
2. **Train LoRA**: Train the actual LoRA model using the dataset

### Step 1: Generate Dataset

1. **Add the "Generate LoRA Dataset" node** to your workflow
2. **Connect your images** to the `images` input (supports lists of ImageArtifact/ImageUrlArtifact)
3. **Configure dataset settings**:
   - `generate_captions`: Enable AI-powered captioning (requires GT_CLOUD_API_KEY)
   - `agent_prompt`: Customize the captioning prompt if needed
   - `image_resolution`: Set training resolution (512 or 1024)
   - `dataset_folder`: Choose where to save the dataset
4. **Run the node** to generate your training dataset

### Step 2: Train LoRA

1. **Add the "Train LoRA" node** to your workflow
2. **Connect the dataset config** from the Generate Dataset node to `dataset_config_path`
3. **Configure training parameters**:
   - `flux_model`: Choose your FLUX.1 model variant
   - `output_dir`: Where to save the trained LoRA
   - `output_name`: Name for your LoRA model
   - `learning_rate`: Training learning rate (default: 1e-6)
   - `max_train_epochs`: Number of training epochs (default: 10)
   - `network_dim`: LoRA network dimension (default: 4)
4. **Run the node** to train your LoRA

## 📋 Node Parameters

### Generate LoRA Dataset Node

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `images` | List[ImageArtifact] | Input images for training | Required |
| `generate_captions` | Boolean | Use AI to generate captions | True |
| `agent` | Agent | Custom agent for captioning | None (uses GPT-4.1-mini) |
| `agent_prompt` | String | Prompt for caption generation | "Describe this image..." |
| `captions` | List[String] | Manual captions (if not generating) | [] |
| `image_resolution` | Integer | Training resolution | 1024 |
| `dataset_folder` | String | Output dataset directory | Required |

### Train LoRA Node

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_family` | String | Model family to train | "FLUX.1" |
| `flux_model` | String | Specific FLUX.1 model | "black-forest-labs/FLUX.1-dev" |
| `dataset_config_path` | String | Path to dataset TOML config | Required |
| `output_dir` | String | Output directory for trained LoRA | Required |
| `output_name` | String | Name for the LoRA model | "my_flux_lora" |
| `learning_rate` | Float | Training learning rate | 1e-6 |
| `max_train_epochs` | Integer | Maximum training epochs | 10 |
| `network_dim` | Integer | LoRA network dimension | 4 |
| `network_alpha` | Float | LoRA network alpha | 1e-3 |
| `mixed_precision` | String | Precision mode (bf16/fp16/no) | "bf16" |
| `fp8_base` | Boolean | Use fp8 quantization | True |
| `highvram` | Boolean | High VRAM mode | True |

## 🎨 Use Cases

### Custom Style Training
Train LoRAs to replicate specific artistic styles, photography techniques, or visual aesthetics.

### Character/Object Training
Create LoRAs for specific characters, objects, or subjects that can be consistently generated.

### Fine-tuning Workflows
Integrate LoRA training into larger AI content creation pipelines.

### Research and Experimentation
Rapid prototyping and testing of custom model adaptations.

## 🔧 Advanced Configuration

### Training Parameters

- **Learning Rate**: Controls how quickly the model learns (1e-6 to 1e-4 typical range)
- **Network Dimension**: Higher values capture more detail but require more VRAM
- **Epochs**: More epochs = longer training but potentially better results
- **Mixed Precision**: bf16 recommended for modern GPUs, fp16 for older hardware

### Memory Optimization

- **fp8_base**: Reduces VRAM usage by quantizing base model to fp8
- **highvram**: Optimizes for high VRAM GPUs (24GB+)
- **gradient_checkpointing**: Trades compute for memory (automatically enabled)

## 🛠️ Example Workflow

Here is an example flow that demonstrates the complete LoRA training process:

![Example Flow](./images/example_flow.png)

This workflow shows:
1. Loading training images
2. Generating a dataset with AI captions
3. Training a LoRA model
4. Using the trained LoRA for inference

## 🔍 Troubleshooting

### Common Issues

#### "Missing GT_CLOUD_API_KEY"
**Solution**: Configure your Griptape Cloud API key in Settings > API Keys & Secrets, or disable automatic captioning and provide manual captions.

#### "CUDA out of memory"
**Solutions**:
- Reduce `network_dim` (try 2 or 1)
- Enable `fp8_base` quantization
- Reduce `image_resolution` to 512
- Reduce batch size in dataset config

#### "Model not found in HuggingFace cache"
**Solution**: The model will be automatically downloaded on first use. Ensure you have sufficient disk space and internet connectivity.

#### Training appears stuck
**Solution**: Check the console logs for detailed progress. Training can take 30 minutes to several hours depending on dataset size and parameters.

### Debug Mode

Check the Griptape Nodes logs for detailed information about the training process, including:
- Dataset generation progress
- Model download status
- Training metrics and loss values
- Memory usage information

## 📄 Technical Details

### Dependencies

The library includes comprehensive ML dependencies:
- PyTorch 2.8.0 with CUDA support
- Transformers 4.54.1 for model handling
- Diffusers 0.32.1 for FLUX.1 integration
- Accelerate 1.6.0 for distributed training
- Various optimizers (Lion, Prodigy, ScheduleFree)
- SafeTensors for secure model serialization

### Training Framework

Built on the industry-standard Kohya sd-scripts framework with:
- Automatic mixed precision training
- Gradient checkpointing for memory efficiency
- Advanced optimizers and schedulers
- Comprehensive logging and monitoring

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/griptape-nodes-lora-training-library/issues)
- **Griptape Community**: [Griptape Discord](https://discord.gg/griptape)
- **Documentation**: [Griptape Nodes Docs](https://github.com/griptape-ai/griptape-nodes)

## 🔗 Related Projects

- [Griptape Framework](https://github.com/griptape-ai/griptape)
- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes)
- [Kohya sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [FLUX.1 Models](https://huggingface.co/black-forest-labs)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ for the Griptape community
