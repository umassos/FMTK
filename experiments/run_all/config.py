
device="cuda:0"
Train=False #set it to False for inference only
backbones={
    'papageip': {
        'model_type': 'papagei',
        'model_name': 'papagei_p',
        'model_config':{
            'in_channels':1, 
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            },
    },
    'papageis': {
        'model_type': 'papagei',
        'model_name': 'papagei_s',
        'model_config':{
            'in_channels':1,
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            'n_experts': 3
            }
    },
    'papageissvri': {
        'model_type': 'papagei',
        'model_name': 'papagei_s_svri',
        'model_config':{
            'in_channels':1,
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
        }
    },
    'chronostiny': {
        'model_type': 'chronos',
        'model_name': 'tiny',
    },
    'chronosmini': {
        'model_type': 'chronos',
        'model_name': 'mini',
    },
    'chronossmall': {
        'model_type': 'chronos',
        'model_name': 'small',
    },
    'chronosbase': {
        'model_type': 'chronos',
        'model_name': 'base',
    },
    'chronoslarge': {
        'model_type': 'chronos',
        'model_name': 'large',
    },
    'momentbase':{
        'model_type':'moment',
        'model_name':'base',
    },
    'momentsmall':{
        'model_type':'moment',
        'model_name':'small',
    },
    'momentlarge':{
        'model_type':'moment',
        'model_name':'large',
    },
    # ── VLM backbones ──────────────────────────────────────────────────
    'moondream': {
        'model_type': 'moondream',
        'model_name': 'moondream',
        'model_id': 'vikhyatk/moondream2',
    },
    'llava-1.5-7b': {
        'model_type': 'llava',
        'model_name': 'llava-1.5-7b',
        'model_id': 'llava-hf/llava-1.5-7b-hf',
    },
    'llava-1.5-13b': {
        'model_type': 'llava',
        'model_name': 'llava-1.5-13b',
        'model_id': 'llava-hf/llava-1.5-13b-hf',
    },
    'llava-v1.6-13b': {
        'model_type': 'llava',
        'model_name': 'llava-v1.6-13b',
        'model_id': 'llava-hf/llava-v1.6-vicuna-13b-hf',
    },
    'qwen-0.5B': {
        'model_type': 'qwen',
        'model_name': 'qwen-0.5B',
        'model_id': 'Qwen/Qwen2-0.5B-Instruct',
    },

    'qwen-2B': {
        'model_type': 'qwen',
        'model_name': 'qwen-2B',
        'model_id': 'Qwen/Qwen2-VL-2B-Instruct',
    },
    'qwen-3B': {
        'model_type': 'qwen',
        'model_name': 'qwen-3B',
        'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
    },
    'qwen-7B': {
        'model_type': 'qwen',
        'model_name': 'qwen-7B',
        'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
    },
    'phi-3.5-vision-instruct': {
        'model_type': 'phi',
        'model_name': 'phi',
        'model_id': 'microsoft/Phi-3.5-vision-instruct',
    },
    'phi-vllm': {
        'model_type': 'phi_vllm',
        'model_name': 'phi3.5-vision',
        'model_id': 'microsoft/Phi-3.5-vision-instruct',
        'model_config': {
            'max_new_tokens': 64,
            'gpu_memory_utilization': 0.75,
            'max_model_len': 2048,
        },
    },
    'molmo': {
        'model_type': 'molmo',
        'model_name': 'molmo',
        'model_id': 'allenai/Molmo-7B-D-0924',
    },
    'llama-vision': {
        'model_type': 'llama_vision',
        'model_name': 'llama-vision',
        'model_id': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
    },
    'minicpm': {
        'model_type': 'minicpm',
        'model_name': 'minicpm',
        'model_id': 'openbmb/MiniCPM-V-2_6',
    },
    'minicpm-2b': {
        'model_type': 'minicpm',
        'model_name': 'minicpm-2b',
        'model_id': 'openbmb/MiniCPM-V-2',
    },
    # ── Vision backbones ───────────────────────────────────────────────
    'dinosmall': {
        'model_type': 'dinov2',
        'model_name': 'small',
    },
    'dinobase': {
        'model_type': 'dinov2',
        'model_name': 'base',
    },
    'dinolarge': {
        'model_type': 'dinov2',
        'model_name': 'large',
    },
    'dinogiant': {
        'model_type': 'dinov2',
        'model_name': 'giant',
    },
    # patch-token variants required for spatial decoders (depth, segmentation)
    'dinosmall-patch': {
        'model_type': 'dinov2',
        'model_name': 'small',
        'model_config': {'return_all_tokens': True},
    },
    'dinobase-patch': {
        'model_type': 'dinov2',
        'model_name': 'base',
        'model_config': {'return_all_tokens': True},
    },
    'dinolarge-patch': {
        'model_type': 'dinov2',
        'model_name': 'large',
        'model_config': {'return_all_tokens': True},
    },
    'dinogiant-patch': {
        'model_type': 'dinov2',
        'model_name': 'giant',
        'model_config': {'return_all_tokens': True},
    },
    'maebase': {
        'model_type': 'mae',
        'model_name': 'base',
    },
    'maelarge': {
        'model_type': 'mae',
        'model_name': 'large',
    },
    'maehuge': {
        'model_type': 'mae',
        'model_name': 'huge',
    },
    'swintiny': {
        'model_type': 'swin',
        'model_name': 'tiny',
    },
    'swinsmall': {
        'model_type': 'swin',
        'model_name': 'small',
    },
    'swinbase': {
        'model_type': 'swin',
        'model_name': 'base',
    },
    'swinlarge': {
        'model_type': 'swin',
        'model_name': 'large',
    },
    'vgg11': {
        'model_type': 'vgg',
        'model_name': 'vgg11',
    },
    'vgg13': {
        'model_type': 'vgg',
        'model_name': 'vgg13',
    },
    'vgg16': {
        'model_type': 'vgg',
        'model_name': 'vgg16',
    },
    'vgg19': {
        'model_type': 'vgg',
        'model_name': 'vgg19',
    },
    # ── LLM (text-only) backbones ──────────────────────────────────────
    'llama-3.1-8b': {
        'model_type': 'llama_text',
        'model_name': 'llama-3.1-8b',
        'model_config': {'max_new_tokens': 10},
    },
    'llama-3.2-3b': {
        'model_type': 'llama_text',
        'model_name': 'llama-3.2-3b',
        'model_config': {'max_new_tokens': 10},
    },
    'mistral-7b': {
        'model_type': 'mistral_text',
        'model_name': 'mistral-7b',
        'model_config': {'max_new_tokens': 10},
    },
    'phi3-mini': {
        'model_type': 'phi3_text',
        'model_name': 'phi3-mini',
        'model_config': {'max_new_tokens': 10},
    },
    'qwen2.5-0.5b': {
        'model_type': 'qwen_text',
        'model_name': 'qwen2.5-0.5b',
        'model_config': {'max_new_tokens': 10},
    },
    'qwen2.5-1.5b': {
        'model_type': 'qwen_text',
        'model_name': 'qwen2.5-1.5b',
        'model_config': {'max_new_tokens': 10},
    },
    'qwen2.5-3b': {
            'model_type': 'qwen_text',
            'model_name': 'qwen2.5-3b',
            'model_config': {'max_new_tokens': 10},
        },
    'qwen2.5-7b': {
        'model_type': 'qwen_text',
        'model_name': 'qwen2.5-7b',
        'model_config': {'max_new_tokens': 10},
    },
    }
decoders={
    'ridge_regression':{
        'decoder_type': 'ridge',
    },
    'svm_class':{
        'decoder_type': 'svm',
    },
    'mlp_momentsmall_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_momentbase_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_momentlarge_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':256,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronosmini_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':384,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronossmall_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronosbase_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_chronoslarge_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_papageis_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },
    'mlp_papageip_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,       
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':5,'hidden_dim':128}
        }
    },

    'mlp_momentlarge_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_momentbase_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_momentsmall_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':256,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronosmini_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':384,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronossmall_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronosbase_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronoslarge_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_papageis_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_papageip_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,       
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_gesture_class':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':8,'hidden_dim':128}
        }
    },
    'mlp_chronostiny_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':513*256,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronosmini_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':513*384,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronossmall_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':513*512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronosbase_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':513*768,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_chronoslarge_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':513*1024,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_momentlarge_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':64*1024,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_momentbase_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':64*768,'output_dim':192,'dropout':0.1}
        }       
    },
    
    'mlp_momentsmall_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':64*512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_papageis_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }       
    },
    'mlp_papageip_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }
    },
    'mlp_papageissvri_forecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':192,'dropout':0.1}
        }
    },

    'mlp_momentlarge_illnessforecasting':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':64*1024,'output_dim':36,'dropout':0.1}
        }       
    },

    'mlp_momentlarge_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_momentbase_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_momentsmall_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronostiny_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':256,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronosmini_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':384,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronossmall_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronosbase_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':768,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_chronoslarge_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':1024,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_papageis_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },
    'mlp_papageip_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,       
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},    
        }
    },
    'mlp_papageissvri_regression':{
        'decoder_type': 'mlp',
        'decoder_config':{
            'device': device,
            'cfg':{'input_dim':512,'output_dim':1,'hidden_dim':128},
        }
    },

    # ── Vision decoders (linear, 10-class classification) ──────────────
    'linear_dinosmall_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 384, 'output_dim': 10},
        }
    },
    'linear_dinobase_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 10},
        }
    },
    'linear_dinolarge_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'output_dim': 10},
        }
    },
    'linear_dinogiant_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1536, 'output_dim': 10},
        }
    },
    'linear_maebase_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 10},
        }
    },
    'linear_maelarge_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'output_dim': 10},
        }
    },
    'linear_maehuge_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1280, 'output_dim': 10},
        }
    },
    'linear_swintiny_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 10},
        }
    },
    'linear_swinsmall_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 10},
        }
    },
    'linear_swinbase_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'output_dim': 10},
        }
    },
    'linear_swinlarge_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1536, 'output_dim': 10},
        }
    },
    'linear_vgg_imgclass10': {
        'decoder_type': 'linear',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 4096, 'output_dim': 10},
        }
    },

    # ── Vision decoders (linear semantic segmentation, VOC12) ─────────────
    # target_size=448, patch_size=14 → grid_size=32, num_classes=21
    'linseg_dinosmall_voc': {
        'decoder_type': 'linear_seg',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 384, 'output_dim': 21, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224, 'ignore_index': 255},
        }
    },
    'linseg_dinobase_voc': {
        'decoder_type': 'linear_seg',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 21, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224, 'ignore_index': 255},
        }
    },
    'linseg_dinolarge_voc': {
        'decoder_type': 'linear_seg',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'output_dim': 21, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224, 'ignore_index': 255},
        }
    },
    'linseg_dinogiant_voc': {
        'decoder_type': 'linear_seg',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1536, 'output_dim': 21, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224, 'ignore_index': 255},
        }
    },

    # ── Vision decoders (monocular depth, NYU Depth V2) ───────────────────
    # target_size=224, patch_size=14 → grid_size=16
    'monodepth_dinosmall': {
        'decoder_type': 'monocular_depth',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 384, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224},
        }
    },
    'monodepth_dinobase': {
        'decoder_type': 'monocular_depth',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224},
        }
    },
    'monodepth_dinolarge': {
        'decoder_type': 'monocular_depth',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224},
        }
    },
    'monodepth_dinogiant': {
        'decoder_type': 'monocular_depth',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1536, 'height': 16, 'width': 16, 'pixel_height': 224, 'pixel_width': 224},
        }
    },

    # ── Vision decoders (spatial count regression) ─────────────────────
    'spatialcount_dinobase': {
        'decoder_type': 'spatial_count',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 768, 'output_dim': 1, 'hidden_dim': 128},
        }
    },
    'spatialcount_dinolarge': {
        'decoder_type': 'spatial_count',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 1024, 'output_dim': 1, 'hidden_dim': 128},
        }
    },
    'spatialcount_dinosmall': {
        'decoder_type': 'spatial_count',
        'decoder_config': {
            'device': device,
            'cfg': {'input_dim': 384, 'output_dim': 1, 'hidden_dim': 128},
        }
    },
}
encoders={
    'linear':{
        'encoder_type': 'linear',
        'encoder_config':{'num_channels':3,'new_num_channels':1}
    }
}
adapters={
    'lora':{
        'adapter_type': 'lora',
        'adapter_config':{'r':64,
                'lora_alpha':32,
                'target_modules':["q", "v"],
                'lora_dropout':0.05}
        },
    'lora_vlm':{
        'adapter_type': 'lora',
        'adapter_config':{'r':16,
                'lora_alpha':32,
                'target_modules':["q_proj", "v_proj"],
                'lora_dropout':0.05}
        },
}

datasets={
    'PPG-data': 
    {
        'dataset_path': '../../dataset/PPG-data',
        'dataset_type': 'PPG-data',
        'seq_len': 512
    },
    'ecg5000':{
        'dataset_path': '../../dataset/ECG5000',
        'dataset_type': 'ECG5000',
    },
    'ETTh1':{
        'dataset_path': '../../dataset/ETTh1',
        'dataset_type': 'ETTh1',   
    },
    'UWaveGestureLibraryAll':{
        'dataset_path': '../../dataset/UWaveGestureLibraryAll',
        'dataset_type': 'UWaveGestureLibraryAll',
        'seq_len': 512},
    'ecl':{
        'dataset_path': '../../dataset/ElectricityLoad-data',
        'dataset_type': 'ecl',},
    'weather':{
        'dataset_path': '../../dataset/Weather',
        'dataset_type': 'weather',},
    'traffic':{
        'dataset_path': '../../dataset/Traffic',
        'dataset_type': 'traffic',},
    'exchange':{    
        'dataset_path': '../../dataset/Exchange',
        'dataset_type': 'exchange',},
    'illness':{
        'dataset_path': '../../dataset/ILLNESS',
        'dataset_type': 'illness',
    },

    # ── VLM datasets ──────────────────────────────────────────────────
    # Paths are relative to experiments/run_all/ (where inference_pipeline.py runs).
    'vlm_activity_recognition': {
        'dataset_path': '../../dataset/vlm/activity_recognition',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_crowd_counting': {
        'dataset_path': '../../dataset/vlm/crowd_counting',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_gesture_recognition': {
        'dataset_path': '../../dataset/vlm/gesture_recognition',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_image_classification': {
        'dataset_path': '../../dataset/vlm/image_classification',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_object_detection': {
        'dataset_path': '../../dataset/vlm/object_detection',
        'dataset_type': 'vlm',
        'json_file': 'annotations.json',
    },
    'vlm_ocr': {
        'dataset_path': '../../dataset/vlm/ocr',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_scene_classification': {
        'dataset_path': '../../dataset/vlm/scene_classification',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_traffic_classification': {
        'dataset_path': '../../dataset/vlm/traffic_classification',
        'dataset_type': 'vlm',
        'json_file': 'labels.json',
    },
    'vlm_vqa': {
        'dataset_path': '../../dataset/vlm/vqa',
        'dataset_type': 'vlm',
        'json_file': 'val.json',
        'image_subdir': 'val2014',
    },

    # ── Vision datasets ────────────────────────────────────────────────
    'VOC12': {
        'dataset_path': '/work/pi_shenoy_umass_edu/kgudipaty/datasets/PASCAL-VOC',
        'dataset_type': 'VOC12',
        'target_size': 224,
    },
    'NYUDepthV2': {
        'dataset_path': '/work/pi_shenoy_umass_edu/kgudipaty/datasets/nyu-depth-v2',
        'dataset_type': 'NYUDepthV2',
        'target_size': 224,
        'max_depth': 10.0,
        'normalize_depth': True,
    },
    'EuroSAT': {
        'dataset_path': '/work/pi_shenoy_umass_edu/kgudipaty/datasets/EuroSAT',
        'dataset_type': 'EuroSAT',
    },
    'CIFAR10': {
        'dataset_path': '../../dataset/cifar-10',
        'dataset_type': 'CIFAR10',
    },
    'ShanghaiTech': {
        'dataset_path': '/work/pi_shenoy_umass_edu/kgudipaty/datasets/ShanghaiTech',
        'dataset_type': 'ShanghaiTech',
    },
    # ── LLM (text-only) datasets ───────────────────────────────────────
    'sst2':          {'dataset_type': 'sst2',        'max_samples': 500},
    'ag_news':       {'dataset_type': 'ag_news',     'max_samples': 500},
    'conll2003':     {'dataset_type': 'conll2003',   'max_samples': 500},
    'squad':         {'dataset_type': 'squad',       'max_samples': 500},
    'cnn_dailymail': {'dataset_type': 'cnn_dailymail','max_samples': 200},
    'flores':        {'dataset_type': 'flores',      'max_samples': 200, 'src_lang': 'fra_Latn', 'tgt_lang': 'eng_Latn'},
    'gsm8k':         {'dataset_type': 'gsm8k',       'max_samples': 200},
    'humaneval':     {'dataset_type': 'humaneval',   'max_samples': 164},
    'hellaswag':     {'dataset_type': 'hellaswag',   'max_samples': 500},
    'fever':         {'dataset_type': 'fever',       'max_samples': 500},
    }

        
tasks = {
    # 'diasbp': {
    #     'task_type': 'regression',
    #     'datasets': ['PPG-data'],
    #     'label': 'diasbp',
    #      'train': Train,
    #      'pipelines':[
            # {
            # 'backbone':'momentlarge',
            # 'paths':[
                    # {'decoder':'ridge_regression','parts_to_train':['decoder']},
                    # {'decoder':'mlp_momentlarge_regression','parts_to_train':['decoder'],'path':'diasbp_momentlarge_mlp'},
                    # {'decoder':'mlp_momentlarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
                    # {'decoder':'mlp_momentlarge_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
                    # {'decoder':'mlp_momentlarge_regression','adapter':'lora','parts_to_train':['decoder','adapter'],'path':'diasbp_momentlarge_mlp_lora'},
                    # ]},
            # {
            # 'backbone':'momentbase',
            # 'paths':[
            #         {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_momentbase_regression','parts_to_train':['decoder'],'path':'diasbp_momentbase_mlp'},
            #         {'decoder':'mlp_momentbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         {'decoder':'mlp_momentbase_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
            #         {'decoder':'mlp_momentbase_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
            #         ]},
            # {
            # 'backbone':'momentsmall',
            # 'paths':[
                    # {'decoder':'ridge_regression','parts_to_train':['decoder']},
                    # {'decoder':'mlp_momentsmall_regression','parts_to_train':['decoder'],'path':'diasbp_momentsmall_mlp'},
                    # {'decoder':'mlp_momentsmall_regression','encoder':'linear','parts_to_train':['decoder','encoder'],'path':'diasbp_momentsmall_mlp_mlp'},
                    # {'decoder':'mlp_momentsmall_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter'],'path':'diasbp_momentsmall_mlp_mlp_lora'},
                    # {'decoder':'mlp_momentsmall_regression','adapter':'lora','parts_to_train':['decoder','adapter'],'path':'diasbp_momentsmall_mlp_lora'},
                    # ]},
            # {
            # 'backbone':'chronostiny',
            # 'paths':[
                    # {'decoder':'ridge_regression','parts_to_train':['decoder']},
                    # {'decoder':'mlp_chronostiny_regression','parts_to_train':['decoder'],'path':'diasbp_chronostiny_mlp'},
                    # {'decoder':'mlp_chronostiny_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
                    # ]},
            # {'backbone':'chronosmini',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_chronosmini_regression','parts_to_train':['decoder'],'path':'diasbp_chronosmini_mlp'},
            #         # {'decoder':'mlp_chronosmini_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {'backbone':'chronossmall',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_chronossmall_regression','parts_to_train':['decoder'],'path':'diasbp_chronossmall_mlp'},
            #         # {'decoder':'mlp_chronossmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {'backbone':'chronosbase',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_chronosbase_regression','parts_to_train':['decoder'],'path':'diasbp_chronosbase_mlp'},
            #         # {'decoder':'mlp_chronosbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {'backbone':'chronoslarge',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_chronoslarge_regression','parts_to_train':['decoder'],'path':'diasbp_chronoslarge_mlp'},
            #         # {'decoder':'mlp_chronoslarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {
            # 'backbone':'papageis',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_papageis_regression','parts_to_train':['decoder'],'path':'diasbp_papageis_mlp'},
            #         # {'decoder':'mlp_papageis_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {
            # 'backbone':'papageip',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_papageip_regression','parts_to_train':['decoder'],'path':'diasbp_papageip_mlp'},
            #         # {'decoder':'mlp_papageip_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]},
            # {
            # 'backbone':'papageissvri',
            # 'paths':[
            #         # {'decoder':'ridge_regression','parts_to_train':['decoder']},
            #         {'decoder':'mlp_papageissvri_regression','parts_to_train':['decoder'],'path':'diasbp_papageissvri_mlp'},
            #         # {'decoder':'mlp_papageissvri_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
            #         ]           
            # }
    #         ],
    #     'inference_config': {
    #         'batch_size': [1,2,4,6,8],
    #         'shuffle':False
    #         },    
    #     'train_config':{
    #         'batch_size': 32,
    #         'shuffle':False,
    #         'epochs':50,
    #         'lr':1e-2,
    #     },
    # },
    # 'sysbp': {
    #     'task_type': 'regression',
    #     'datasets': ['PPG-data'],
    #     'label': 'sysbp',
    #     'train': Train,
    #     'pipelines':[
    #         {
    #         'backbone':'momentlarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentlarge_regression','parts_to_train':['decoder'],'path':'sysbp_momentlarge_mlp'},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentlarge_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentbase_regression','parts_to_train':['decoder'],'path':'sysbp_momentbase_mlp'},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentbase_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentsmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentsmall_regression','parts_to_train':['decoder'],'path':'sysbp_momentsmall_mlp'},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentsmall_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'chronostiny',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronostiny_regression','parts_to_train':['decoder'],'path':'sysbp_chronostiny_mlp'},
    #                 # {'decoder':'mlp_chronostiny_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosmini',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosmini_regression','parts_to_train':['decoder'],'path':'sysbp_chronosmini_mlp'},
    #                 # {'decoder':'mlp_chronosmini_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronossmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronossmall_regression','parts_to_train':['decoder'],'path':'sysbp_chronossmall_mlp'},
    #                 # {'decoder':'mlp_chronossmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosbase_regression','parts_to_train':['decoder'],'path':'sysbp_chronosbase_mlp'},
    #                 # {'decoder':'mlp_chronosbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronoslarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronoslarge_regression','parts_to_train':['decoder'],'path':'sysbp_chronoslarge_mlp'},
    #                 # {'decoder':'mlp_chronoslarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageis',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageis_regression','parts_to_train':['decoder'],'path':'sysbp_papageis_mlp'},
    #                 # {'decoder':'mlp_papageis_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageip',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageip_regression','parts_to_train':['decoder'],'path':'sysbp_papageip_mlp'},
    #                 # {'decoder':'mlp_papageip_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageissvri',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageissvri_regression','parts_to_train':['decoder'],'path':'sysbp_papageissvri_mlp'},
    #                 # {'decoder':'mlp_papageissvri_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]           
    #         }
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },    
    #     'train_config':{
    #         'batch_size': 32,
    #         'shuffle':False,
    #         'epochs':50,
    #         'lr':1e-2
    #     },
    # },
    # 'heartrate': {
    #     'task_type': 'regression',
    #     'datasets': ['PPG-data'],
    #     'label': 'hr',
    #     'train': Train,
    #     'pipelines':[
    #         {
    #         'backbone':'momentlarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentlarge_regression','parts_to_train':['decoder'],'path':'heartrate_momentlarge_mlp'},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentlarge_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentbase_regression','parts_to_train':['decoder'],'path':'heartrate_momentbase_mlp'},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentbase_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentsmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentsmall_regression','parts_to_train':['decoder'],'path':'heartrate_momentsmall_mlp'},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentsmall_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'chronostiny',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronostiny_regression','parts_to_train':['decoder'],'path':'heartrate_chronostiny_mlp'},
    #                 # {'decoder':'mlp_chronostiny_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosmini',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosmini_regression','parts_to_train':['decoder'],'path':'heartrate_chronosmini_mlp'},
    #                 # {'decoder':'mlp_chronosmini_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronossmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronossmall_regression','parts_to_train':['decoder'], 'path':'heartrate_chronossmall_mlp'},
    #                 # {'decoder':'mlp_chronossmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosbase_regression','parts_to_train':['decoder'],'path':'heartrate_chronosbase_mlp'},
    #                 # {'decoder':'mlp_chronosbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronoslarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronoslarge_regression','parts_to_train':['decoder'],'path':'heartrate_chronoslarge_mlp'},
    #                 # {'decoder':'mlp_chronoslarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageis',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageis_regression','parts_to_train':['decoder'],'path':'heartrate_papageis_mlp'},
    #                 # {'decoder':'mlp_papageis_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageip',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageip_regression','parts_to_train':['decoder'],'path':'heartrate_papageip_mlp'},
    #                 # {'decoder':'mlp_papageip_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageissvri',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageissvri_regression','parts_to_train':['decoder'],'path':'heartrate_papageissvri_mlp'},
    #                 # {'decoder':'mlp_papageissvri_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]           
    #         }
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },    
    #     'train_config':{
    #         'batch_size': 32,
    #         'shuffle':False,
    #         'epochs':50,
    #         'lr':1e-2
    #     },
    # },
    # 'ecgclass': {
    # 'task_type': 'classification',
    # 'datasets': ['ecg5000'],
    # 'train': Train,
    # 'pipelines':[
            # {
            # 'backbone':'momentlarge',
            # 'paths':[
            #     {'decoder':'mlp_momentlarge_class','parts_to_train':['decoder'],'path':'ecgclass_momentlarge_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'momentsmall',
            # 'paths':[
            #     {'decoder':'mlp_momentsmall_class','parts_to_train':['decoder'],'path':'ecgclass_momentsmall_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'momentbase',
            # 'paths':[
            #     # {'decoder':'mlp_momentbase_class','parts_to_train':['decoder'],'path':'ecgclass_momentbase_mlp'},
            #     # {'decoder':'mlp_momentbase_class','encoder':'linear','parts_to_train':['decoder','encoder'],'path':'ecgclass_momentbase_mlp_linear'},
            #     # {'decoder':'mlp_momentbase_class','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter'],'path':'ecgclass_momentbase_mlp_linear_lora'},
            #     {'decoder':'mlp_momentbase_class','adapter':'lora','parts_to_train':['decoder','adapter'],'path':'ecgclass_momentbase_mlp_lora'},
            #     ]
            # },
            # {
            # 'backbone':'chronostiny',
            # 'paths':[
            #     {'decoder':'mlp_chronostiny_class','parts_to_train':['decoder'],'path':'ecgclass_chronostiny_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronosmini',
            # 'paths':[
            #     {'decoder':'mlp_chronosmini_class','parts_to_train':['decoder'],'path':'ecgclass_chronosmini_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronossmall',
            # 'paths':[
            #     {'decoder':'mlp_chronossmall_class','parts_to_train':['decoder'],'path':'ecgclass_chronossmall_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronosbase',
            # 'paths':[
            #     {'decoder':'mlp_chronosbase_class','parts_to_train':['decoder'],'path':'ecgclass_chronosbase_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronoslarge',
            # 'paths':[
            #     {'decoder':'mlp_chronoslarge_class','parts_to_train':['decoder'],'path':'ecgclass_chronoslarge_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'papageis',
            # 'paths':[
            #     {'decoder':'mlp_papageis_class','parts_to_train':['decoder'],'path':'ecgclass_papageis_mlp'},
            #     ]   
            # },
            # {
            # 'backbone':'papageip',
            # 'paths':[
            #     {'decoder':'mlp_papageip_class','parts_to_train':['decoder'],'path':'ecgclass_papageip_mlp'},
            #     ]   
            # },
            # {
            # 'backbone':'papageissvri',
            # 'paths':[       
            #     {'decoder':'mlp_papageissvri_class','parts_to_train':['decoder'],'path':'ecgclass_papageissvri_mlp'},
            #     ]   
            # }
    #         ],    
    # 'inference_config': {
    #     'batch_size': 1,
    #     'shuffle':False
    #     },
    # 'train_config': {
    #     'batch_size': 32,
    #     'shuffle':True,
    #     'epochs':50,
    #     'lr':1e-2,
    #     },
    # },
    # 'gestureclass': {
    #     'task_type': 'classification',
    #     'datasets': ['UWaveGestureLibraryAll'],
    #     'train': Train,
    #     'pipelines':[
            # {
            # 'backbone':'momentlarge',
            # 'paths':[
            #     {'decoder':'mlp_momentlarge_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentlarge_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'momentsmall',
            # 'paths':[
            #     {'decoder':'mlp_momentsmall_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentsmall_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'momentbase',
            # 'paths':[
                # {'decoder':'mlp_momentbase_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentbase_mlp'},
                # {'decoder':'mlp_momentbase_gesture_class','encoder':'linear','parts_to_train':['decoder','encoder'],'path':'gestureclass_momentbase_mlp_linear'},
                # {'decoder':'mlp_momentbase_gesture_class','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter'],'path':'gestureclass_momentbase_mlp_linear_lora'},
            #     {'decoder':'mlp_momentbase_gesture_class','adapter':'lora','parts_to_train':['decoder','adapter'],'path':'gestureclass_momentbase_mlp_lora'},
            #     ]
            # },
            # {
            # 'backbone':'chronostiny',
            # 'paths':[
            #     {'decoder':'mlp_chronostiny_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronostiny_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronosmini',
            # 'paths':[
            #     {'decoder':'mlp_chronosmini_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronosmini_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronossmall',
            # 'paths':[
            #     {'decoder':'mlp_chronossmall_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronossmall_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronosbase',
            # 'paths':[
            #     {'decoder':'mlp_chronosbase_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronosbase_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'chronoslarge',
            # 'paths':[
            #     {'decoder':'mlp_chronoslarge_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronoslarge_mlp'},
            #     ]
            # },
            # {
            # 'backbone':'papageis',
            # 'paths':[
            #     {'decoder':'mlp_papageis_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageis_mlp'},
            #     ]   
            # },
            # {
            # 'backbone':'papageip',
            # 'paths':[
            #     {'decoder':'mlp_papageip_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageip_mlp'},
            #     ]   
            # },
            # {
            # 'backbone':'papageissvri',
            # 'paths':[       
            #     {'decoder':'mlp_papageissvri_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageissvri_mlp'},
            #     ]   
            # }
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 32,
    #         'shuffle':True,
    #         'epochs':50,
    #         'lr':1e-2,
    #     },
    # },
    # 'etth1fore':{
    #     'task_type': 'forecasting',
    #     'datasets': ['ETTh1'],
    #     'train': Train,
    #     'pipelines':[
    #     {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'etth1fore_chronosbase_mlp'}]},
    #     {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'etth1fore_chronossmall_mlp'}]},
    #     {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'etth1fore_chronosmini_mlp'}]},
    #     {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'etth1fore_chronostiny_mlp'}]},
    #     {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'etth1fore_chronoslarge_mlp'}]},
    #     {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'etth1fore_momentlarge_mlp'}]},
    #     {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'etth1fore_momentbase_mlp'}]},
    #     {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'etth1fore_momentsmall_mlp'}]},
    #     {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'etth1fore_papageis_mlp'}]},
    #     {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'etth1fore_papageip_mlp'}]},
    #     {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'etth1fore_papageissvri_mlp'}]},
    #     ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 8,
    #         'shuffle':True,
    #         'epochs':1,
    #         'lr':1e-4,
    #         },
    # },
    # 'weatherfore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['weather'],
    #     'train': Train,
    #     'pipelines':[
    #     {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'weatherfore_chronosbase_mlp'}]},
    #     {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'weatherfore_chronossmall_mlp'}]},
    #     {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'weatherfore_chronosmini_mlp'}]},
    #     {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'weatherfore_chronostiny_mlp'}]},
    #     {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'weatherfore_chronoslarge_mlp'}]},
    #     {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'weatherfore_momentlarge_mlp'}]},
    #     {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'weatherfore_momentbase_mlp'}]},
    #     {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'weatherfore_momentsmall_mlp'}]},
    #     {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'weatherfore_papageis_mlp'}]},
    #     {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'weatherfore_papageip_mlp'}]},
    #     {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'weatherfore_papageissvri_mlp'}]},
    #     ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 8,
    #         'shuffle':True,
    #         'epochs':1,
    #         'lr': 5e-5,
    #         },
    # },
    # 'exchangefore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['exchange'],
    #     'train': Train,
    #     'pipelines':[
    #             {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'exchangefore_chronosbase_mlp'}]},
    #             {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'], 'path':'exchangefore_chronossmall_mlp'}]},
    #             {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'], 'path':'exchangefore_chronosmini_mlp'}]},
    #             {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'], 'path':'exchangefore_chronostiny_mlp'}]},
    #             {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'], 'path':'exchangefore_chronoslarge_mlp'}]},
    #             {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'exchangefore_momentlarge_mlp'}]},
    #             {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'exchangefore_momentbase_mlp'}]},
    #             {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'exchangefore_momentsmall_mlp'}]},
    #             {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'exchangefore_papageis_mlp'}]},
    #             {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'exchangefore_papageip_mlp'}]},
    #             {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'exchangefore_papageissvri_mlp'}]},
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False         
    #         },
    #     'train_config': {
    #         'batch_size': 8,
    #         'shuffle':True,     
    #         'epochs':1,
    #         'lr': 5e-5
    #         },
    # }, 
    # 'Illness forecasting': {
    #     'task_type': 'forecasting',
    #     'datasets': ['illness'],
    #     'train': Train,
    #     'pipelines':[{
    #         'backbone':'momentlarge',            
    #         'paths':[
    #             {'decoder':'mlp_momentlarge_illnessforecasting','parts_to_train':['decoder']},
    #             ]
    #         }],
    #     'inference_config': {
    #         'batch_size': 8,
    #         'shuffle':False         
    #         },
    #     'train_config': {
    #         'batch_size': 4,
    #         'shuffle':True,     
    #         'epochs':1,
    #         'lr': 5e-5,
    #         },
    # }, 
    # 'eclfore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['ecl'],
    #     'train': Train,
    #     'pipelines':[
    #             {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'eclfore_chronosbase_mlp'}]},
    #             {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'], 'path':'eclfore_chronossmall_mlp'}]},
    #             {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'], 'path':'eclfore_chronosmini_mlp'}]},
    #             {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'], 'path':'eclfore_chronostiny_mlp'}]},
    #             {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'], 'path':'eclfore_chronoslarge_mlp'}]},
    #             {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'eclfore_momentlarge_mlp'}]},
    #             {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'eclfore_momentbase_mlp'}]},
    #             {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'eclfore_momentsmall_mlp'}]},
    #             {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'eclfore_papageis_mlp'}]},
    #             {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'eclfore_papageip_mlp'}]},
    #             {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'eclfore_papageissvri_mlp'}]},
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 4,
    #         'shuffle':True,
    #         'epochs':1,
    #         'lr': 5e-5,
    #         },
    # },
    # 'trafficfore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['traffic'],
    #     'train': Train,
    #     'pipelines':[
    #             {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'trafficfore_chronosbase_mlp'}]},
    #             {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'trafficfore_chronossmall_mlp'}]},
    #             {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'trafficfore_chronosmini_mlp'}]},
    #             {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'trafficfore_chronostiny_mlp'}]},
    #             {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'trafficfore_chronoslarge_mlp'}]},
    #             {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'trafficfore_momentlarge_mlp'}]},
    #             {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'trafficfore_momentbase_mlp'}]},
    #             {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'trafficfore_momentsmall_mlp'}]},
    #             {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'trafficfore_papageis_mlp'}]},
    #             {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'trafficfore_papageip_mlp'}]},
    #             {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'trafficfore_papageissvri_mlp'}]},
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 2,
    #         'shuffle':True,
    #         'epochs':1,
    #         'lr': 5e-5,
    #         },
    # },

    # # # ── VLM tasks ──────────────────────────────────────────────────────
    # 'vlm_crowd': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'crowd',
    #     'datasets': ['vlm_crowd_counting'],
    #     'train': False,
    #     'parser': 'parse_crowd_label',
    #     'evaluator': 'evaluate_crowd',
    #     'train_ratio': 0.8,
    #     'train_config': {'batch_size': 1, 'shuffle': True, 'lr': 1e-4, 'epochs': 2,'max_samples': 100},
    #     'pipelines': [
    #         # {'backbone': 'llama-vision', 'paths': [{}]},
    #         # {'backbone': 'minicpm', 'paths': [{}]},
    #         # {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },
    # 'vlm_scene': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'scene',
    #     'datasets': ['vlm_scene_classification'],
    #     'train': False,
    #     'parser': 'parse_scene_label',
    #     'evaluator': 'evaluate_scene',
    #     'pipelines': [
    #         {'backbone': 'llama-vision', 'paths': [{}]},
    #         {'backbone': 'minicpm', 'paths': [{}]},
    #         {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },
    # 'vlm_ocr': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'ocr',
    #     'datasets': ['vlm_ocr'],
    #     'train': Train,
    #     'parser': 'parse_ocr_digit',
    #     'evaluator': 'evaluate_ocr',
    #     'train_ratio': 0.8,
    #     'train_config': {'batch_size': 1, 'shuffle': True, 'lr': 1e-4, 'epochs': 2,'max_samples': 100},
    #     'pipelines': [
            # {'backbone': 'llama-vision', 'paths': [{}]},
            # {'backbone': 'minicpm-2b', 'paths':
            #  [{},
            #   ]},
            # {'backbone': 'qwen-2B', 'paths':
            #  [{},
            #   {'adapter': 'lora_vlm', 'parts_to_train': ['adapter'], 'path': 'vlm_ocr_qwen_lora'},
            #   ]},
            # {'backbone': 'qwen-3B', 'paths':
            #  [{},
            # #   {'adapter': 'lora_vlm', 'parts_to_train': ['adapter'], 'path': 'vlm_ocr_qwen7B_lora'},
            #   ]},
            # {'backbone': 'qwen-7B', 'paths':
            #  [{},
            # #   {'adapter': 'lora_vlm', 'parts_to_train': ['adapter'], 'path': 'vlm_ocr_qwen14B_lora'},
            #   ]},
            # {'backbone': 'molmo', 'paths': 
            #  [{}]},
            # {'backbone': 'phi-3.5-vision-instruct', 'paths': [
            #     {},
                # {'adapter': 'lora_vlm', 'parts_to_train': ['adapter'], 'path': 'vlm_ocr_phi_lora'},
            # ]},
            # {'backbone': 'phi-vllm', 'paths': [
            #     {},
            #     # {'adapter': 'lora_vlm', 'path': 'vlm_ocr_phi_lora'},
            # ]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False,'max_samples': 100},
    # },
    # 'vlm_vqa': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'vqa',
    #     'datasets': ['vlm_vqa'],
    #     'train': False,
    #     'parser': 'parse_vqa_label',
    #     'evaluator': 'evaluate_vqa',
    #     'train_ratio': 0.8,
    #     'pipelines': [
            # {'backbone': 'llama-vision', 'paths': [{}]},
            # {'backbone': 'minicpm', 'paths': [{}]},
            # {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'train_config': {'batch_size': 1, 'shuffle': True, 'lr': 1e-4, 'epochs': 2,'max_samples': 100},
    #     'inference_config': {'batch_size': 1, 'shuffle': False,'max_samples': 100},
    # },
    # 'vlm_traffic': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'traffic',
    #     'datasets': ['vlm_traffic_classification'],
    #     'train': False,
    #     'parser': 'parse_traffic_label',
    #     'evaluator': 'evaluate_substring_match',
    #     'train_ratio': 0.8,
    #     'pipelines': [
            # {'backbone': 'llama-vision', 'paths': [{}]},
            # {'backbone': 'minicpm', 'paths': [{}]},
            # {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'train_config': {'batch_size': 1, 'shuffle': True, 'lr': 1e-4, 'epochs': 2,'max_samples': 100},
    #     'inference_config': {'batch_size': 1, 'shuffle': False,'max_samples': 100},
    # },
    # 'vlm_gesture': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'gesture',
    #     'datasets': ['vlm_gesture_recognition'],
    #     'train': False,
    #     'parser': 'parse_gesture_label',
    #     'evaluator': 'evaluate_substring_match',
    #     'pipelines': [
    #         {'backbone': 'llama-vision', 'paths': [{}]},
    #         {'backbone': 'minicpm', 'paths': [{}]},
    #         {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },
    # 'vlm_activity': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'activity',
    #     'datasets': ['vlm_activity_recognition'],
    #     'train': False,
    #     'parser': 'parse_activity_label',
    #     'evaluator': 'evaluate_substring_match',
    #     'pipelines': [
    #         {'backbone': 'llama-vision', 'paths': [{}]},
    #         {'backbone': 'minicpm', 'paths': [{}]},
    #         {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },
    # 'vlm_object_detection': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'object_detection',
    #     'datasets': ['vlm_object_detection'],
    #     'train': False,
    #     'parser': 'parse_object_detection_label',
    #     'evaluator': 'evaluate_object_detection',
    #     'pipelines': [
    #         {'backbone': 'llama-vision', 'paths': [{}]},
    #         {'backbone': 'minicpm', 'paths': [{}]},
    #         {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },
    # 'vlm_image_classification': {
    #     'task_type': 'vlm',
    #     'vlm_task_key': 'image_classification',
    #     'datasets': ['vlm_image_classification'],
    #     'train': False,
    #     'parser': 'parse_classification_label',
    #     'evaluator': 'evaluate_image_classification',
    #     'pipelines': [
    #         {'backbone': 'llama-vision', 'paths': [{}]},
    #         {'backbone': 'minicpm', 'paths': [{}]},
    #         {'backbone': 'molmo', 'paths': [{}]},
    #         {'backbone': 'phi-3.5-vision-instruct', 'paths': [{}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    # },

    # ── Vision tasks ───────────────────────────────────────────────────
    # 'eurosat_imgclass': {
    #     'task_type': 'classification',
    #     'datasets': ['EuroSAT'],
    #     'train': Train,
    #     'pipelines': [
    #         {'backbone': 'dinosmall',  'paths': [{'decoder': 'linear_dinosmall_imgclass10',  'parts_to_train': ['decoder'], 'path': 'eurosatclass_dinosmall_linear'}]},
    #         {'backbone': 'dinobase',   'paths': [{'decoder': 'linear_dinobase_imgclass10',   'parts_to_train': ['decoder'], 'path': 'eurosatclass_dinobase_linear'}]},
    #         {'backbone': 'dinolarge',  'paths': [{'decoder': 'linear_dinolarge_imgclass10',  'parts_to_train': ['decoder'], 'path': 'eurosatclass_dinolarge_linear'}]},
            # {'backbone': 'dinogiant',  'paths': [{'decoder': 'linear_dinogiant_imgclass10',  'parts_to_train': ['decoder'], 'path': 'eurosatclass_dinogiant_linear'}]},
            # {'backbone': 'maebase',    'paths': [{'decoder': 'linear_maebase_imgclass10',    'parts_to_train': ['decoder'], 'path': 'eurosatclass_maebase_linear'}]},
            # {'backbone': 'maelarge',   'paths': [{'decoder': 'linear_maelarge_imgclass10',   'parts_to_train': ['decoder'], 'path': 'eurosat_maelarge_linear'}]},
            # {'backbone': 'maehuge',    'paths': [{'decoder': 'linear_maehuge_imgclass10',    'parts_to_train': ['decoder'], 'path': 'eurosat_maehuge_linear'}]},
            # {'backbone': 'swintiny',   'paths': [{'decoder': 'linear_swintiny_imgclass10',   'parts_to_train': ['decoder'], 'path': 'eurosatclass_swintiny_linear'}]},
            # {'backbone': 'swinsmall',  'paths': [{'decoder': 'linear_swinsmall_imgclass10',  'parts_to_train': ['decoder'], 'path': 'eurosatclass_swinsmall_linear'}]},
            # {'backbone': 'swinbase',   'paths': [{'decoder': 'linear_swinbase_imgclass10',   'parts_to_train': ['decoder'], 'path': 'eurosatclass_swinbase_linear'}]},
            # {'backbone': 'swinlarge',  'paths': [{'decoder': 'linear_swinlarge_imgclass10',  'parts_to_train': ['decoder'], 'path': 'eurosatclass_swinlarge_linear'}]},
            # {'backbone': 'vgg16',      'paths': [{'decoder': 'linear_vgg_imgclass10',        'parts_to_train': ['decoder'], 'path': 'eurosat_vgg16_linear'}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    #     'train_config': {
    #         'batch_size': 32,
    #         'shuffle': True,
    #         'epochs': 20,
    #         'lr': 1e-3,
    #         'scheduler': {'type': 'cosine', 'T_max': 10, 'eta_min': 0},
    #     },
    # },
    # 'cifar10_imgclass': {
    #     'task_type': 'classification',
    #     'datasets': ['CIFAR10'],
    #     'train': Train,
    #     'pipelines': [
    #         {'backbone': 'dinosmall', 'paths': [{'decoder': 'linear_dinosmall_imgclass10', 'parts_to_train': ['decoder'], 'path': 'cifar10_dinosmall_linear'}]},
    #         {'backbone': 'dinobase',  'paths': [{'decoder': 'linear_dinobase_imgclass10',  'parts_to_train': ['decoder'], 'path': 'cifar10_dinobase_linear'}]},
    #         {'backbone': 'dinolarge', 'paths': [{'decoder': 'linear_dinolarge_imgclass10', 'parts_to_train': ['decoder'], 'path': 'cifar10_dinolarge_linear'}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    #     'train_config': {
    #         'batch_size': 32,
    #         'shuffle': False,
    #         'epochs': 50,
    #         'lr': 1e-2,
    #     },
    # },
    # 'shanghaitech_count': {
    #     'task_type': 'regression',
    #     'datasets': ['ShanghaiTech'],
    #     'train': Train,
    #     'pipelines': [
    #         {'backbone': 'dinosmall', 'paths': [{'decoder': 'spatialcount_dinosmall', 'parts_to_train': ['decoder'], 'path': 'shanghaitech_dinosmall_spatialcount'}]},
    #         {'backbone': 'dinobase',  'paths': [{'decoder': 'spatialcount_dinobase',  'parts_to_train': ['decoder'], 'path': 'shanghaitech_dinobase_spatialcount'}]},
    #         {'backbone': 'dinolarge', 'paths': [{'decoder': 'spatialcount_dinolarge', 'parts_to_train': ['decoder'], 'path': 'shanghaitech_dinolarge_spatialcount'}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    #     'train_config': {
    #         'batch_size': 32,
    #         'shuffle': True,
    #         'epochs': 100,
    #         'lr': 1e-5,
    #         'scheduler': {'type': 'cosine', 'T_max': 10, 'eta_min': 0},
    #     },
    # },
    # 'nyu_depth': {
    #     'task_type': 'regression',
    #     'datasets': ['NYUDepthV2'],
    #     'train': Train,
    #     'pipelines': [
    #         # {'backbone': 'dinosmall-patch',  'paths': [{'decoder': 'monodepth_dinosmall',  'parts_to_train': ['decoder'], 'path': 'nyudepth_dinosmall_monocular'}]},
    #         {'backbone': 'dinobase-patch',   'paths': [{'decoder': 'monodepth_dinobase',   'parts_to_train': ['decoder'], 'path': 'nyudepth_dinobase_monocular'}]},
    #         # {'backbone': 'dinolarge-patch',  'paths': [{'decoder': 'monodepth_dinolarge',  'parts_to_train': ['decoder'], 'path': 'nyudepth_dinolarge_monocular'}]},
    #         # {'backbone': 'dinogiant-patch',  'paths': [{'decoder': 'monodepth_dinogiant',  'parts_to_train': ['decoder'], 'path': 'nyudepth_dinogiant_monocular'}]},
    #     ],
    #     'inference_config': {'batch_size': 1, 'shuffle': False},
    #     'train_config': {
    #         'batch_size': 16,
    #         'shuffle': True,
    #         'epochs': 10,
    #         'lr': 1e-3,
    #         'scheduler': {'type': 'cosine', 'T_max': 10, 'eta_min': 0},
    #         'use_cache': True,
    #     },
    # },
    'voc_seg': {
        'task_type': 'segmentation',
        'datasets': ['VOC12'],
        'train': Train,
        'pipelines': [
            # {'backbone': 'dinosmall-patch',  'paths': [{'decoder': 'linseg_dinosmall_voc',  'parts_to_train': ['decoder'], 'path': 'vocseg_dinosmall_linsemseg'}]},
            {'backbone': 'dinobase-patch',   'paths': [{'decoder': 'linseg_dinobase_voc',   'parts_to_train': ['decoder'], 'path': None}]},
            # {'backbone': 'dinolarge-patch',  'paths': [{'decoder': 'linseg_dinolarge_voc',  'parts_to_train': ['decoder'], 'path': 'vocseg_dinolarge_linsemseg'}]},
            # {'backbone': 'dinogiant-patch',  'paths': [{'decoder': 'linseg_dinogiant_voc',  'parts_to_train': ['decoder'], 'path': 'vocseg_dinogiant_linsemseg'}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
        'train_config': {
            'batch_size': 8,
            'shuffle': True,
            'epochs': 10,
            'lr': 1e-3,
            'scheduler': {'type': 'cosine', 'T_max': 10, 'eta_min': 0},
            'use_cache': True,
        },
    },

    # # ── LLM (text-only) tasks ──────────────────────────────────────────
    # 'llm_sst2': {
    #     'task_type': 'sentiment',
    #     'datasets': ['sst2'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
            # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
            # {'backbone': 'qwen2.5-0.5b', 'paths': [{'path': ''}]},
            # {'backbone': 'qwen2.5-1.5b',   'paths': [{'path': ''}]},
            # {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
            # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
            # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
            # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_ag_news': {
    #     'task_type': 'text_classification',
    #     'datasets': ['ag_news'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-0.5b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_conll2003': {
    #     'task_type': 'ner',
    #     'datasets': ['conll2003'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-0.5b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_squad': {
    #     'task_type': 'qa',
    #     'datasets': ['squad'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_cnn_dailymail': {
    #     'task_type': 'summarization',
    #     'datasets': ['cnn_dailymail'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_flores': {
    #     'task_type': 'translation',
    #     'datasets': ['flores'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_gsm8k': {
    #     'task_type': 'math_reasoning',
    #     'datasets': ['gsm8k'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_humaneval': {
    #     'task_type': 'code_generation',
    #     'datasets': ['humaneval'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_hellaswag': {
    #     'task_type': 'reading_comprehension',
    #     'datasets': ['hellaswag'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
    # 'llm_fever': {
    #     'task_type': 'fact_verification',
    #     'datasets': ['fever'],
    #     'train': False,
    #     'train_config': {'batch_size': 1, 'shuffle': False},
    #     'inference_config': {'batch_size': [1,2,4,6,8,10], 'shuffle': False},
    #     'pipelines': [
    #         # {'backbone': 'phi3-mini',    'paths': [{'path': ''}]},
    #         # {'backbone': 'llama-3.1-8b', 'paths': [{'path': ''}]},
    #         # {'backbone': 'mistral-7b',   'paths': [{'path': ''}]},
    #         {'backbone': 'qwen2.5-3b',   'paths': [{'path': ''}]},
    #         # {'backbone': 'qwen2.5-7b',   'paths': [{'path': ''}]},
    #     ],
    # },
}
#segmentation, regression, classification
#eurosat, cifar,
#vgg, resnet, swin, mae, dinov2

log_file= "combined_metrics.csv"
vlm_log_file= "vlm_metrics.csv"
vlm_csv_columns = [
    "model_name", "dataset_name", "device", "model_load_duration_sec",
    "gpu_load_memory_mb", "avg_cpu_memory_usage_mb", "avg_cpu_usage_percent",
    "avg_gpu_usage_percent", "avg_gpu_memory_usage_mb", "total_prompt_tokens",
    "total_generated_tokens", "ttft_ms", "avg_latency_ms", "throughput_tps",
    "accuracy", "total_time", "num_samples", "gpu_name",
]
