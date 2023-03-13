import tensorflow as tf

def load_layers_from_ckpt(path):
    ckpt = tf.train.load_checkpoint(path)
    ckpt_layers = {k: ckpt.get_tensor(k) for k in ckpt.get_variable_to_shape_map() \
                   if 'Momentum' not in k and 'ExponentialMovingAverage' not in k}
    return ckpt_layers

def convert_layer_name_to_ckpt_layer_name(layer_name):
    if 'cls' in layer_name or 'link' in layer_name:
        ckpt_name = layer_name.split('_', 2)
        ckpt_name = '_'.join(ckpt_name[:-1]) + '/score_from_' + ckpt_name[-1]
    else:
        ckpt_name = layer_name.split('_')
        if len(ckpt_name) == 1:
            ckpt_name = ckpt_name[0]
        else:
            ckpt_name = ckpt_name[0] + '/' + layer_name       
    return ckpt_name

def get_model_layers(model):
    layers = list(set(
        [layer.split('.')[0] for layer in model.get_weight_paths().keys()]
    ))
    return layers

def set_weights_and_biases(model, ckpt_path):
    layers = get_model_layers(model)
    ckpt_layers = load_layers_from_ckpt(ckpt_path)
    for layer_name in layers:
        ckpt_layer_name = convert_layer_name_to_ckpt_layer_name(layer_name) 
        model.get_layer(layer_name).set_weights([
            ckpt_layers[f'{ckpt_layer_name}/weights'],
            ckpt_layers[f'{ckpt_layer_name}/biases']
        ])