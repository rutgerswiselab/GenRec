# GenRec
Large Language Model for Generative Recommendation

## Install dependencies

    
    pip install -r requirements.txt
    
    
### Training (`rec.py`)

    
    python rec.py \
        --base_model 'decapoda-research/llama-7b-hf' \
        --data_path './moives' \
        --output_dir './checkpoint'
    
    
    
### Inference (`generate.py`)


    
    python generate.py \
        --load_8bit \
        --base_model 'decapoda-research/llama-7b-hf' \
        --lora_weights './checkpoint/movies'


This project is implemented based on alpaca-lora (https://github.com/tloen/alpaca-lora)