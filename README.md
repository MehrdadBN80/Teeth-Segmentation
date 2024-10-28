
## Model Architecture

The model uses a modified U-Net structure with encoder and decoder blocks:
- **Encoder**: Captures context via a contracting path.
- **Decoder**: Reconstructs segmentation masks by upsampling and fusing encoder features with skip connections.
- **Loss Functions**: Includes Dice Loss, Focal Loss, and a combined loss function to improve segmentation accuracy.

### Important Model Components:
1. **ConvolutionBlock**: Basic convolutional block with options for encoder and decoder modes.
2. **EncoderBlock** and **DecoderBlock**: Custom blocks that make up the U-Net architecture.
3. **Generator**: Main U-Net structure with encoder and decoder paths.
4. **Model**: Wrapper class for the generator, making it the main segmentation model.

## Training and Evaluation

The model is trained using a combination of **Dice Loss** and **Focal Loss** to handle class imbalance effectively. The Dice Score, a widely used metric in segmentation tasks, is calculated during training to evaluate the model’s performance.

### Evaluation Metric:
- **Dice Score**: Measures the overlap between predicted and actual segmentations, calculated at each epoch for tracking progress.

### Training Process:
- Train and validate using Dice Score.
- The best model based on Dice Score is saved in `model_checkpoints/best_generator_model.pth`.

### Data Augmentation and Preprocessing
- **Data Augmentation**: Essential for improving model generalization on this relatively small dataset.
- **Data Cleaning**: Address any potential data poisoning issues before training.

## Code Structure

The main files and folders in this repository are:

- `teeth-damaged-detection.ipynb`: Jupyter Notebook with code for model training and evaluation.
- `requirements.txt`: List of required dependencies.

