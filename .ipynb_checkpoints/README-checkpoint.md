# GenRec
Large Language Model for Generative Recommendation

## Install dependencies

    ```bash
    pip install -r requirements.txt
    ```
    
### Training (`rec.py`)

    ```bash
    python rec.py \
        --base_model 'decapoda-research/llama-7b-hf' \
        --data_path './moives' \
        --output_dir './checkpoint'
    ```
    
    
### Inference (`generate.py`)


    ```bash
    python generate.py \
        --load_8bit \
        --base_model 'decapoda-research/llama-7b-hf' \
        --lora_weights './checkpoint/movies'
    ```