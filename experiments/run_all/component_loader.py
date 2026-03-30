def get_model_class(model_type):
    if model_type == 'papagei':
        from fmtk.components.backbones.papagei import PapageiModel
        return PapageiModel
    elif model_type == 'chronos':
        from fmtk.components.backbones.chronos import ChronosModel
        return ChronosModel
    elif model_type == 'moment':
        from fmtk.components.backbones.moment import MomentModel
        return MomentModel
    elif model_type == 'llava':
        from fmtk.components.backbones.llava import LlavaModel
        return LlavaModel
    elif model_type == 'llama_vision':
        from fmtk.components.backbones.llama_vision import LlamaVisionModel
        return LlamaVisionModel
    elif model_type == 'minicpm':
        from fmtk.components.backbones.minicpm import MinicpmModel
        return MinicpmModel
    elif model_type == 'molmo':
        from fmtk.components.backbones.molmo import MolmoModel
        return MolmoModel
    elif model_type == 'moondream':
        from fmtk.components.backbones.moondream import MoondreamModel
        return MoondreamModel
    elif model_type == 'phi':
        from fmtk.components.backbones.phi import PhiModel
        return PhiModel
    elif model_type == 'phi_vllm':
        from fmtk.components.backbones.phi import PhiVLLMModel
        return PhiVLLMModel
    elif model_type == 'qwen':
        from fmtk.components.backbones.qwen import QwenModel
        return QwenModel
    elif model_type == 'dinov2':
        from fmtk.components.backbones.dinov2 import DinoV2Model
        return DinoV2Model
    elif model_type == 'mae':
        from fmtk.components.backbones.mae import MAEModel
        return MAEModel
    elif model_type == 'swin':
        from fmtk.components.backbones.swin import SwinModel
        return SwinModel
    elif model_type == 'vgg':
        from fmtk.components.backbones.vgg import VGGModel
        return VGGModel
    # ── LLM (text-only) backbones ──────────────────────────────────────
    elif model_type == 'llama_text':
        from fmtk.components.backbones.llama import LlamaModel
        return LlamaModel
    elif model_type == 'mistral_text':
        from fmtk.components.backbones.mistral import MistralModel
        return MistralModel
    elif model_type == 'phi3_text':
        from fmtk.components.backbones.phi3 import Phi3Model
        return Phi3Model
    elif model_type == 'qwen_text':
        from fmtk.components.backbones.qwen import QwenTextModel
        return QwenTextModel
    raise ValueError(f"Unknown model type: {model_type}")

def get_decoder_class(task_type,decoder_type):
    if task_type=='regression':
        if decoder_type == 'ridge':
            from fmtk.components.decoders.regression.ridge import RidgeDecoder
            return RidgeDecoder
        elif decoder_type == 'mlp':
            from fmtk.components.decoders.regression.mlp import MLPDecoder
            return MLPDecoder
        elif decoder_type == 'spatial_count':
            from fmtk.components.decoders.regression.spatial_count import SpatialCountDecoder
            return SpatialCountDecoder
        elif decoder_type == 'monocular_depth':
            from fmtk.components.decoders.regression.monocular_depth import MonocularDepthDecoder
            return MonocularDepthDecoder
    elif task_type=='segmentation':
        if decoder_type == 'linear_seg':
            from fmtk.components.decoders.segmentation.LinearSemanticSegmenter import LinearSemanticSegmenter
            return LinearSemanticSegmenter
    elif task_type=='classification':
        if decoder_type == 'logistic':
            from fmtk.components.decoders.classification.logisticregression import LogisticDecoder
            return LogisticDecoder
        elif decoder_type == 'random_forest':
            from fmtk.components.decoders.classification.randomforest import RandomForestDecoder
            return RandomForestDecoder
        elif decoder_type == 'svm':
                from fmtk.components.decoders.classification.svm import SVMDecoder
                return SVMDecoder
        elif decoder_type == 'knn':
            from fmtk.components.decoders.classification.knn import KNNDecoder
            return KNNDecoder
        elif decoder_type == 'mlp':
            from fmtk.components.decoders.classification.mlp import MLPDecoder
            return MLPDecoder
        elif decoder_type == 'linear':
            from fmtk.components.decoders.classification.linear import LinearDecoder
            return LinearDecoder
    elif task_type=='forecasting':
        if decoder_type == 'mlp':
            from fmtk.components.decoders.forecasting.mlp import MLPDecoder
            return MLPDecoder
    return None

def get_encoder_class(encoder_type):
    if encoder_type == 'linear':
        from fmtk.components.encoders.diff import LinearChannelCombiner
        return LinearChannelCombiner
    return None

def get_adapter_class(adapter_type):
    if adapter_type == 'lora':
        from peft import LoraConfig
        return LoraConfig
    return None
