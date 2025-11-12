def get_model_class(model_type):
    if model_type == 'papagei':
        from timeseries.components.backbones.papagei import PapageiModel
        return PapageiModel
    elif model_type == 'chronos':
        from timeseries.components.backbones.chronos import ChronosModel
        return ChronosModel
    elif model_type == 'moment':
        from timeseries.components.backbones.moment import MomentModel
        return MomentModel
    raise ValueError(f"Unknown dataset type: {model_type}")

def get_decoder_class(task_type,decoder_type):
    if task_type=='regression':
        if decoder_type == 'ridge':
            from timeseries.components.decoders.regression.ridge import RidgeDecoder
            return RidgeDecoder
        elif decoder_type == 'mlp':
            from timeseries.components.decoders.regression.mlp import MLPDecoder
            return MLPDecoder
    elif task_type=='classification':
        if decoder_type == 'logistic':
            from timeseries.components.decoders.classification.logisticregression import LogisticDecoder
            return LogisticDecoder
        elif decoder_type == 'random_forest':
            from timeseries.components.decoders.classification.randomforest import RandomForestDecoder
            return RandomForestDecoder
        elif decoder_type == 'svm':
                from timeseries.components.decoders.classification.svm import SVMDecoder
                return SVMDecoder
        elif decoder_type == 'knn':
            from timeseries.components.decoders.classification.knn import KNNDecoder
            return KNNDecoder
        elif decoder_type == 'mlp':
            from timeseries.components.decoders.classification.mlp import MLPDecoder
            return MLPDecoder
    elif task_type=='forecasting':
        if decoder_type == 'mlp':
            from timeseries.components.decoders.forecasting.mlp import MLPDecoder
            return MLPDecoder
    return None

def get_encoder_class(encoder_type):
    if encoder_type == 'linear':
        from timeseries.components.encoders.diff import LinearChannelCombiner
        return LinearChannelCombiner
    return None

def get_adapter_class(adapter_type):
    if adapter_type == 'lora':
        from peft import LoraConfig
        return LoraConfig
    return None
