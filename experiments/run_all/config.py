
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
    'phi': {
        'model_type': 'phi',
        'model_name': 'phi',
        'model_id': 'microsoft/Phi-3.5-vision-instruct',
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
    }
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
        }
}

datasets={
    'PPG-data': 
    {
        'dataset_path': '../../dataset/PPG-data',
        'dataset_type': 'PPG-data',
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
        'dataset_type': 'UWaveGestureLibraryAll',},
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
    }

        
tasks = {
    # 'diasbp': {
    #     'task_type': 'regression',
    #     'datasets': ['PPG-data'],
    #     'label': 'diasbp',
    #      'train': Train,
    #      'pipelines':[
    #         {
    #         'backbone':'momentlarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentlarge_regression','parts_to_train':['decoder'],'path':'diasbp_momentlarge_mlp'},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentlarge_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentlarge_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentbase_regression','parts_to_train':['decoder'],'path':'diasbp_momentbase_mlp'},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 # {'decoder':'mlp_momentbase_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter']},
    #                 # {'decoder':'mlp_momentbase_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'momentsmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_momentsmall_regression','parts_to_train':['decoder'],'path':'diasbp_momentsmall_mlp'},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','parts_to_train':['decoder','encoder'],'path':'diasbp_momentsmall_mlp_mlp'},
    #                 # {'decoder':'mlp_momentsmall_regression','encoder':'linear','adapter':'lora','parts_to_train':['decoder','encoder','adapter'],'path':'diasbp_momentsmall_mlp_mlp_lora'},
    #                 # {'decoder':'mlp_momentsmall_regression','adapter':'lora','parts_to_train':['decoder','adapter']},
    #                 ]},
    #         {
    #         'backbone':'chronostiny',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronostiny_regression','parts_to_train':['decoder'],'path':'diasbp_chronostiny_mlp'},
    #                 # {'decoder':'mlp_chronostiny_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosmini',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosmini_regression','parts_to_train':['decoder'],'path':'diasbp_chronosmini_mlp'},
    #                 # {'decoder':'mlp_chronosmini_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronossmall',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronossmall_regression','parts_to_train':['decoder'],'path':'diasbp_chronossmall_mlp'},
    #                 # {'decoder':'mlp_chronossmall_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronosbase',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronosbase_regression','parts_to_train':['decoder'],'path':'diasbp_chronosbase_mlp'},
    #                 # {'decoder':'mlp_chronosbase_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {'backbone':'chronoslarge',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_chronoslarge_regression','parts_to_train':['decoder'],'path':'diasbp_chronoslarge_mlp'},
    #                 # {'decoder':'mlp_chronoslarge_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageis',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageis_regression','parts_to_train':['decoder'],'path':'diasbp_papageis_mlp'},
    #                 # {'decoder':'mlp_papageis_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageip',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageip_regression','parts_to_train':['decoder'],'path':'diasbp_papageip_mlp'},
    #                 # {'decoder':'mlp_papageip_regression','encoder':'linear','parts_to_train':['decoder','encoder']},
    #                 ]},
    #         {
    #         'backbone':'papageissvri',
    #         'paths':[
    #                 # {'decoder':'ridge_regression','parts_to_train':['decoder']},
    #                 {'decoder':'mlp_papageissvri_regression','parts_to_train':['decoder'],'path':'diasbp_papageissvri_mlp'},
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
    #         {
    #         'backbone':'momentlarge',
    #         'paths':[
    #             {'decoder':'mlp_momentlarge_class','parts_to_train':['decoder'],'path':'ecgclass_momentlarge_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'momentsmall',
    #         'paths':[
    #             {'decoder':'mlp_momentsmall_class','parts_to_train':['decoder'],'path':'ecgclass_momentsmall_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'momentbase',
    #         'paths':[
    #             {'decoder':'mlp_momentbase_class','parts_to_train':['decoder'],'path':'ecgclass_momentbase_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronostiny',
    #         'paths':[
    #             {'decoder':'mlp_chronostiny_class','parts_to_train':['decoder'],'path':'ecgclass_chronostiny_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronosmini',
    #         'paths':[
    #             {'decoder':'mlp_chronosmini_class','parts_to_train':['decoder'],'path':'ecgclass_chronosmini_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronossmall',
    #         'paths':[
    #             {'decoder':'mlp_chronossmall_class','parts_to_train':['decoder'],'path':'ecgclass_chronossmall_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronosbase',
    #         'paths':[
    #             {'decoder':'mlp_chronosbase_class','parts_to_train':['decoder'],'path':'ecgclass_chronosbase_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronoslarge',
    #         'paths':[
    #             {'decoder':'mlp_chronoslarge_class','parts_to_train':['decoder'],'path':'ecgclass_chronoslarge_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'papageis',
    #         'paths':[
    #             {'decoder':'mlp_papageis_class','parts_to_train':['decoder'],'path':'ecgclass_papageis_mlp'},
    #             ]   
    #         },
    #         {
    #         'backbone':'papageip',
    #         'paths':[
    #             {'decoder':'mlp_papageip_class','parts_to_train':['decoder'],'path':'ecgclass_papageip_mlp'},
    #             ]   
    #         },
    #         {
    #         'backbone':'papageissvri',
    #         'paths':[       
    #             {'decoder':'mlp_papageissvri_class','parts_to_train':['decoder'],'path':'ecgclass_papageissvri_mlp'},
    #             ]   
    #         }
    #         ],    
    # 'inference_config': {
    #     'batch_size': 1,
    #     'shuffle':False
    #     },
    # 'train_config': {
    #     'batch_size': 32,
    #     'shuffle':False,
    #     'epochs':50,
    #     'lr':1e-2,
    #     },
    # },
    # 'gestureclass': {
    #     'task_type': 'classification',
    #     'datasets': ['UWaveGestureLibraryAll'],
    #     'train': Train,
    #     'pipelines':[
    #         {
    #         'backbone':'momentlarge',
    #         'paths':[
    #             {'decoder':'mlp_momentlarge_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentlarge_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'momentsmall',
    #         'paths':[
    #             {'decoder':'mlp_momentsmall_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentsmall_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'momentbase',
    #         'paths':[
    #             {'decoder':'mlp_momentbase_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_momentbase_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronostiny',
    #         'paths':[
    #             {'decoder':'mlp_chronostiny_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronostiny_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronosmini',
    #         'paths':[
    #             {'decoder':'mlp_chronosmini_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronosmini_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronossmall',
    #         'paths':[
    #             {'decoder':'mlp_chronossmall_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronossmall_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronosbase',
    #         'paths':[
    #             {'decoder':'mlp_chronosbase_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronosbase_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'chronoslarge',
    #         'paths':[
    #             {'decoder':'mlp_chronoslarge_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_chronoslarge_mlp'},
    #             ]
    #         },
    #         {
    #         'backbone':'papageis',
    #         'paths':[
    #             {'decoder':'mlp_papageis_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageis_mlp'},
    #             ]   
    #         },
    #         {
    #         'backbone':'papageip',
    #         'paths':[
    #             {'decoder':'mlp_papageip_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageip_mlp'},
    #             ]   
    #         },
    #         {
    #         'backbone':'papageissvri',
    #         'paths':[       
    #             {'decoder':'mlp_papageissvri_gesture_class','parts_to_train':['decoder'],'path':'gestureclass_papageissvri_mlp'},
    #             ]   
    #         }
    #         ],
    #     'inference_config': {
    #         'batch_size': 1,
    #         'shuffle':False
    #         },
    #     'train_config': {
    #         'batch_size': 32,
    #         'shuffle':False,
    #         'epochs':50,
    #         'lr':1e-2,
    #     },
    # },
    'etth1fore':{
        'task_type': 'forecasting',
        'datasets': ['ETTh1'],
        'train': Train,
        'pipelines':[
        {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_chronosbase_mlp'}]},
        {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_chronossmall_mlp'}]},
        {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_chronosmini_mlp'}]},
        {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_chronostiny_mlp'}]},
        {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_chronoslarge_mlp'}]},
        {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_momentlarge_mlp'}]},
        {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_momentbase_mlp'}]},
        {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_momentsmall_mlp'}]},
        {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_papageis_mlp'}]},
        {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_papageip_mlp'}]},
        {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'etth1_fore_papageissvri_mlp'}]},
        ],
        'inference_config': {
            'batch_size': 1,
            'shuffle':False
            },
        'train_config': {
            'batch_size': 8,
            'shuffle':True,
            'epochs':1,
            'lr':1e-4,
            },
    },
    # 'weatherfore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['weather'],
    #     'train': Train,
    #     'pipelines':[
    #     {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'weather_fore_chronosbase_mlp'}]},
    #     {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'weather_fore_chronossmall_mlp'}]},
    #     {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'weather_fore_chronosmini_mlp'}]},
    #     {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'weather_fore_chronostiny_mlp'}]},
    #     {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'weather_fore_chronoslarge_mlp'}]},
    #     {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'weather_fore_momentlarge_mlp'}]},
    #     {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'weather_fore_momentbase_mlp'}]},
    #     {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'weather_fore_momentsmall_mlp'}]},
    #     {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'weather_fore_papageis_mlp'}]},
    #     {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'weather_fore_papageip_mlp'}]},
    #     {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'weather_fore_papageissvri_mlp'}]},
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
    'ratefore': {
        'task_type': 'forecasting',
        'datasets': ['exchange'],
        'train': Train,
        'pipelines':[
                {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_chronosbase_mlp'}]},
                {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'], 'path':'exchange_fore_chronossmall_mlp'}]},
                {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'], 'path':'exchange_fore_chronosmini_mlp'}]},
                {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'], 'path':'exchange_fore_chronostiny_mlp'}]},
                {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'], 'path':'exchange_fore_chronoslarge_mlp'}]},
                {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_momentlarge_mlp'}]},
                {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_momentbase_mlp'}]},
                {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_momentsmall_mlp'}]},
                {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_papageis_mlp'}]},
                {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_papageip_mlp'}]},
                {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'exchange_fore_papageissvri_mlp'}]},
            ],
        'inference_config': {
            'batch_size': 1,
            'shuffle':False         
            },
        'train_config': {
            'batch_size': 8,
            'shuffle':True,     
            'epochs':1,
            'lr': 5e-5
            },
    }, 
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
    'elecfore': {
        'task_type': 'forecasting',
        'datasets': ['ecl'],
        'train': Train,
        'pipelines':[
                {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_chronosbase_mlp'}]},
                {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'], 'path':'ecl_fore_chronossmall_mlp'}]},
                {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'], 'path':'ecl_fore_chronosmini_mlp'}]},
                {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'], 'path':'ecl_fore_chronostiny_mlp'}]},
                {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'], 'path':'ecl_fore_chronoslarge_mlp'}]},
                {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_momentlarge_mlp'}]},
                {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_momentbase_mlp'}]},
                {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_momentsmall_mlp'}]},
                {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_papageis_mlp'}]},
                {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_papageip_mlp'}]},
                {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'ecl_fore_papageissvri_mlp'}]},
            ],
        'inference_config': {
            'batch_size': 1,
            'shuffle':False
            },
        'train_config': {
            'batch_size': 4,
            'shuffle':True,
            'epochs':1,
            'lr': 5e-5,
            },
    },
    # 'trafficfore': {
    #     'task_type': 'forecasting',
    #     'datasets': ['traffic'],
    #     'train': Train,
    #     'pipelines':[
    #             {'backbone':'chronosbase','paths':[{'decoder':'mlp_chronosbase_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_chronosbase_mlp'}]},
    #             {'backbone':'chronossmall','paths':[{'decoder':'mlp_chronossmall_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_chronossmall_mlp'}]},
    #             {'backbone':'chronosmini','paths':[{'decoder':'mlp_chronosmini_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_chronosmini_mlp'}]},
    #             {'backbone':'chronostiny','paths':[{'decoder':'mlp_chronostiny_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_chronostiny_mlp'}]},
    #             {'backbone':'chronoslarge','paths':[{'decoder':'mlp_chronoslarge_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_chronoslarge_mlp'}]},
    #             {'backbone':'momentlarge','paths':[{'decoder':'mlp_momentlarge_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_momentlarge_mlp'}]},
    #             {'backbone':'momentbase','paths':[{'decoder':'mlp_momentbase_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_momentbase_mlp'}]},
    #             {'backbone':'momentsmall','paths':[{'decoder':'mlp_momentsmall_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_momentsmall_mlp'}]},
    #             {'backbone':'papageis','paths':[{'decoder':'mlp_papageis_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_papageis_mlp'}]},
    #             {'backbone':'papageip','paths':[{'decoder':'mlp_papageip_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_papageip_mlp'}]},
    #             {'backbone':'papageissvri','paths':[{'decoder':'mlp_papageissvri_forecasting','parts_to_train':['decoder'],'path':'traffic_fore_papageissvri_mlp'}]},
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

    # ── VLM tasks ──────────────────────────────────────────────────────
    'vlm_crowd': {
        'task_type': 'vlm',
        'vlm_task_key': 'crowd',
        'datasets': ['vlm_crowd_counting'],
        'train': False,
        'parser': 'parse_crowd_label',
        'evaluator': 'evaluate_crowd',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_scene': {
        'task_type': 'vlm',
        'vlm_task_key': 'scene',
        'datasets': ['vlm_scene_classification'],
        'train': False,
        'parser': 'parse_scene_label',
        'evaluator': 'evaluate_scene',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_ocr': {
        'task_type': 'vlm',
        'vlm_task_key': 'ocr',
        'datasets': ['vlm_ocr'],
        'train': False,
        'parser': 'parse_ocr_digit',
        'evaluator': 'evaluate_ocr',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_vqa': {
        'task_type': 'vlm',
        'vlm_task_key': 'vqa',
        'datasets': ['vlm_vqa'],
        'train': False,
        'parser': 'parse_vqa_label',
        'evaluator': 'evaluate_vqa',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_traffic': {
        'task_type': 'vlm',
        'vlm_task_key': 'traffic',
        'datasets': ['vlm_traffic_classification'],
        'train': False,
        'parser': 'parse_traffic_label',
        'evaluator': 'evaluate_substring_match',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_gesture': {
        'task_type': 'vlm',
        'vlm_task_key': 'gesture',
        'datasets': ['vlm_gesture_recognition'],
        'train': False,
        'parser': 'parse_gesture_label',
        'evaluator': 'evaluate_substring_match',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_activity': {
        'task_type': 'vlm',
        'vlm_task_key': 'activity',
        'datasets': ['vlm_activity_recognition'],
        'train': False,
        'parser': 'parse_activity_label',
        'evaluator': 'evaluate_substring_match',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_object_detection': {
        'task_type': 'vlm',
        'vlm_task_key': 'object_detection',
        'datasets': ['vlm_object_detection'],
        'train': False,
        'parser': 'parse_object_detection_label',
        'evaluator': 'evaluate_object_detection',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
    'vlm_image_classification': {
        'task_type': 'vlm',
        'vlm_task_key': 'image_classification',
        'datasets': ['vlm_image_classification'],
        'train': False,
        'parser': 'parse_classification_label',
        'evaluator': 'evaluate_image_classification',
        'pipelines': [
            {'backbone': 'llama-vision', 'paths': [{}]},
            {'backbone': 'minicpm', 'paths': [{}]},
            {'backbone': 'molmo', 'paths': [{}]},
            {'backbone': 'phi', 'paths': [{}]},
        ],
        'inference_config': {'batch_size': 1, 'shuffle': False},
    },
}

log_file= "combined_metrics.csv"
vlm_log_file= "vlm_metrics.csv"
vlm_csv_columns = [
    "model_name", "dataset_name", "device", "model_load_duration_sec",
    "gpu_load_memory_mb", "avg_cpu_memory_usage_mb", "avg_cpu_usage_percent",
    "avg_gpu_usage_percent", "avg_gpu_memory_usage_mb", "total_prompt_tokens",
    "total_generated_tokens", "ttft_ms", "avg_latency_ms", "throughput_tps",
    "accuracy", "total_time", "num_samples", "gpu_name",
]
