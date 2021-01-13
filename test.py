import numpy as np
from keras import backend as K
from keras.layers import Concatenate
from keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
from keras.models import Model
from keras.optimizers import Adam


def has_exog(exo_dim):
    return exo_dim > 0


def linear_space(backcast_length, forecast_length, fwd_looking=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / backcast_length
    if fwd_looking:
        ls = ls[backcast_length:]
    else:
        ls = ls[:backcast_length]
    return ls


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)], axis=0)
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)], axis=0)
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, fwd_looking=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)], axis=0))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))


def _r(layer_with_weights, stack_id, share_weights_in_stack, weights):
    # mechanism to restore weights when block share the same weights.
    # only useful when share_weights_in_stack=True.
    if share_weights_in_stack:
        layer_name = layer_with_weights.name.split('/')[-1]
        try:
            reused_weights = weights[stack_id][layer_name]
            return reused_weights
        except KeyError:
            pass
        if stack_id not in weights:
            weights[stack_id] = {}
        weights[stack_id][layer_name] = layer_with_weights
    return layer_with_weights


def create_block(x, e, stack_id, block_id, stack_type, nb_poly,
                 units, backcast_length, forecast_length, share_weights_in_stack, weights, nb_harmonics):

    # register weights (useful when share_weights_in_stack=True)
    def reg(layer):
        return _r(layer, stack_id, share_weights_in_stack, weights)

    # update name (useful when share_weights_in_stack=True)
    def n(layer_name):
        return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

    backcast_ = {}
    forecast_ = {}
    d1 = reg(Dense(units, activation='relu', name=n('d1')))
    d2 = reg(Dense(units, activation='relu', name=n('d2')))
    d3 = reg(Dense(units, activation='relu', name=n('d3')))
    d4 = reg(Dense(units, activation='relu', name=n('d4')))
    if stack_type == 'generic':
        theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
        theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
        backcast = reg(Dense(backcast_length, activation='linear', name=n('backcast')))
        forecast = reg(Dense(forecast_length, activation='linear', name=n('forecast')))
    elif stack_type == 'trend':
        theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
        backcast = Lambda(trend_model, arguments={"is_forecast": False, "backcast_length": backcast_length,
                                                  "forecast_length": forecast_length})
        forecast = Lambda(trend_model, arguments={"is_forecast": True, "backcast_length": backcast_length,
                                                  "forecast_length": forecast_length})
    else:  # 'seasonality'
        if nb_harmonics:
            theta_b = reg(Dense(nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
        else:
            theta_b = reg(Dense(forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
        theta_f = reg(Dense(forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
        backcast = Lambda(seasonality_model,
                          arguments={"is_forecast": False, "backcast_length": backcast_length,
                                     "forecast_length": forecast_length})
        forecast = Lambda(seasonality_model,
                          arguments={"is_forecast": True, "backcast_length": backcast_length,
                                     "forecast_length": forecast_length})
    for k in range(input_dim):
        if has_exog(exo_dim):
            d0 = Concatenate()([x[k]] + [e[ll] for ll in range(exo_dim)])
        else:
            d0 = x[k]
        d1_ = d1(d0)
        d2_ = d2(d1_)
        d3_ = d3(d2_)
        d4_ = d4(d3_)
        theta_f_ = theta_f(d4_)
        theta_b_ = theta_b(d4_)
        backcast_[k] = backcast(theta_b_)
        forecast_[k] = forecast(theta_f_)

    return backcast_, forecast_


if __name__ == '__main__':

    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    units = 256
    input_dim = 1
    exo_dim = 0
    backcast_length = 10
    forecast_length = 2
    stack_types = (TREND_BLOCK, SEASONALITY_BLOCK)
    nb_blocks_per_stack = 3
    thetas_dim = (4, 8)
    share_weights_in_stack = False
    hidden_layer_units = 256
    nb_harmonics = None

    input_shape = (backcast_length, input_dim)
    exo_shape = (backcast_length, exo_dim)
    output_shape = (forecast_length, input_dim)
    weights = {}
    nb_harmonics = nb_harmonics

    x = Input(shape=input_shape, name='input_variable')
    x_ = {}
    for k in range(input_dim):
        x_[k] = Lambda(lambda z: z[..., k])(x)
    e_ = {}
    if has_exog(exo_dim):
        e = Input(shape=exo_shape, name='exos_variables')
        for k in range(exo_dim):
            e_[k] = Lambda(lambda z: z[..., k])(e)
    else:
        e = None
    y_ = {}

    for stack_id in range(len(stack_types)):
        stack_type = stack_types[stack_id]
        nb_poly = thetas_dim[stack_id]
        for block_id in range(nb_blocks_per_stack):
            backcast, forecast = create_block(x_, e_, stack_id, block_id, stack_type, nb_poly,
                                              256,
                                              backcast_length,
                                              forecast_length,
                                              share_weights_in_stack,
                                              weights,
                                              nb_harmonics)
            for k in range(input_dim):
                x_[k] = Subtract()([x_[k], backcast[k]])
                if stack_id == 0 and block_id == 0:
                    y_[k] = forecast[k]
                else:
                    y_[k] = Add()([y_[k], forecast[k]])