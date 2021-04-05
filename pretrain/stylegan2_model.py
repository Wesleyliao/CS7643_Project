#
# ADAPTED FROM REPO Tetratrio/stylegan2_pytorch
# https://github.com/Tetratrio/stylegan2_pytorch/blob/master/stylegan2/models.py
# Modified to be able to load and run existing model.
#
import collections
import copy
import numbers
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _BaseModel(nn.Module):
    """
    Adds some base functionality to models that inherit this class.
    """

    def __init__(self):
        super(_BaseModel, self).__setattr__('kwargs', {})
        super(_BaseModel, self).__setattr__('_defaults', {})
        super(_BaseModel, self).__init__()

    def _update_kwargs(self, **kwargs):
        """
        Update the current keyword arguments. Overrides any
        default values set.
        Arguments:
            **kwargs: Keyword arguments
        """
        self.kwargs.update(**kwargs)

    def _update_default_kwargs(self, **defaults):
        """
        Update the default values for keyword arguments.
        Arguments:
            **defaults: Keyword arguments
        """
        self._defaults.update(**defaults)

    def __getattr__(self, name):
        """
        Try to get the keyword argument for this attribute.
        If no keyword argument of this name exists, try to
        get the attribute directly from this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        Returns:
            value
        """
        try:
            return self.__getattribute__('kwargs')[name]
        except KeyError:
            try:
                return self.__getattribute__('_defaults')[name]
            except KeyError:
                return super(_BaseModel, self).__getattr__(name)

    def __setattr__(self, name, value):
        """
        Try to set the keyword argument for this attribute.
        If no keyword argument of this name exists, set
        the attribute directly for this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
            value
        """
        if name != '__dict__' and (name in self.kwargs or name in self._defaults):
            self.kwargs[name] = value
        else:
            super(_BaseModel, self).__setattr__(name, value)

    def __delattr__(self, name):
        """
        Try to delete the keyword argument for this attribute.
        If no keyword argument of this name exists, delete
        the attribute of this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        """
        deleted = False
        if name in self.kwargs:
            del self.kwargs[name]
            deleted = True
        if name in self._defaults:
            del self._defaults[name]
            deleted = True
        if not deleted:
            super(_BaseModel, self).__delattr__(name)

    def clone(self):
        """
        Create a copy of this model.
        Returns:
            model_copy (nn.Module)
        """
        return copy.deepcopy(self)

    def _get_state_dict(self):
        """
        Delegate function for getting the state dict.
        Should be overridden if state dict has to be
        fetched in abnormal way.
        """
        return self.state_dict()

    def _set_state_dict(self, state_dict):
        """
        Delegate function for loading the state dict.
        Should be overridden if state dict has to be
        loaded in abnormal way.
        """
        self.load_state_dict(state_dict)

    def _serialize(self, half=False):
        """
        Turn model arguments and weights into
        a dict that can safely be pickled and unpickled.
        Arguments:
            half (bool): Save weights in half precision.
                Default value is False.
        """
        state_dict = self._get_state_dict()
        for key in state_dict.keys():
            values = state_dict[key].cpu()
            if torch.is_floating_point(values):
                if half:
                    values = values.half()
                else:
                    values = values.float()
            state_dict[key] = values
        return {
            'name': self.__class__.__name__,
            'kwargs': self.kwargs,
            'state_dict': state_dict,
        }

    @classmethod
    def load(cls, fpath, map_location='cpu'):
        """
        Load a model of this class.
        Arguments:
            fpath (str): File path of saved model.
            map_location (str, int, torch.device): Weights and
                buffers will be loaded into this device.
                Default value is 'cpu'.
        """
        model = load(fpath, map_location=map_location)
        assert isinstance(model, cls), 'Trying to load a `{}` '.format(
            type(model)
        ) + 'model from {}.load()'.format(cls.__name__)
        return model

    def save(self, fpath, half=False):
        """
        Save this model.
        Arguments:
            fpath (str): File path of save location.
            half (bool): Save weights in half precision.
                Default value is False.
        """
        torch.save(self._serialize(half=half), fpath)


def _deserialize(state):
    """
    Load a model from its serialized state.
    Arguments:
        state (dict)
    Returns:
        model (nn.Module): Model that inherits `_BaseModel`.
    """
    state = state.copy()
    name = state.pop('name')
    if name not in globals():
        raise NameError('Class {} is not defined.'.format(state['name']))
    kwargs = state.pop('kwargs')
    state_dict = state.pop('state_dict')
    # Assume every other entry in the state is a serialized
    # keyword argument.
    for key in list(state.keys()):
        kwargs[key] = _deserialize(state.pop(key))
    model = globals()[name](**kwargs)
    model._set_state_dict(state_dict)
    return model


def load(fpath, map_location='cpu'):
    """
    Load a model.
    Arguments:
        fpath (str): File path of saved model.
        map_location (str, int, torch.device): Weights and
            buffers will be loaded into this device.
            Default value is 'cpu'.
    Returns:
        model (nn.Module): Model that inherits `_BaseModel`.
    """
    if map_location is not None:
        map_location = torch.device(map_location)
    return _deserialize(torch.load(fpath, map_location=map_location))


def save(model, fpath, half=False):
    """
    Save a model.
    Arguments:
        model (nn.Module): Wrapped or unwrapped module
            that inherits `_BaseModel`.
        fpath (str): File path of save location.
        half (bool): Save weights in half precision.
            Default value is False.
    """
    unwrap_module(model).save(fpath, half=half)


class Generator(_BaseModel):
    """
    A wrapper class for the latent mapping model
    and synthesis (generator) model.
    Keyword Arguments:
        G_mapping (GeneratorMapping)
        G_synthesis (GeneratorSynthesis)
        dlatent_avg_beta (float): The beta value
            of the exponential moving average
            of the dlatents. This statistic
            is used for truncation of dlatents.
            Default value is 0.995
    """

    def __init__(self, *, G_mapping, G_synthesis, **kwargs):
        super(Generator, self).__init__()
        self._update_default_kwargs(dlatent_avg_beta=0.995)
        self._update_kwargs(**kwargs)

        assert isinstance(
            G_mapping, GeneratorMapping
        ), '`G_mapping` has to be an instance of `model.GeneratorMapping`'
        assert isinstance(
            G_synthesis, GeneratorSynthesis
        ), '`G_synthesis` has to be an instance of `model.GeneratorSynthesis`'
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.register_buffer('dlatent_avg', torch.zeros(self.G_mapping.latent_size))
        self.set_truncation()

    @property
    def latent_size(self):
        return self.G_mapping.latent_size

    @property
    def label_size(self):
        return self.G_mapping.label_size

    def _get_state_dict(self):
        state_dict = OrderedDict()
        self._save_to_state_dict(destination=state_dict, prefix='', keep_vars=False)
        return state_dict

    def _set_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

    def _serialize(self, half=False):
        state = super(Generator, self)._serialize(half=half)
        for name in ['G_mapping', 'G_synthesis']:
            state[name] = getattr(self, name)._serialize(half=half)
        return state

    def set_truncation(self, truncation_psi=None, truncation_cutoff=None):
        """
        Set the truncation of dlatents before they are passed to the
        synthesis model.
        Arguments:
            truncation_psi (float): Beta value of linear interpolation between
                the average dlatent and the current dlatent. 0 -> 100% average,
                1 -> 0% average.
            truncation_cutoff (int, optional): Truncation is only used up until
                this affine layer index.
        """
        layer_psi = None
        if (
            truncation_psi is not None
            and truncation_psi != 1
            and truncation_cutoff != 0
        ):
            layer_psi = torch.ones(len(self.G_synthesis))
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi_mask = torch.arange(len(layer_psi)) < truncation_cutoff
                layer_psi[layer_psi_mask] *= truncation_psi
            layer_psi = layer_psi.view(1, -1, 1)
            layer_psi = layer_psi.to(self.dlatent_avg)
        self.register_buffer('layer_psi', layer_psi)

    def random_noise(self):
        """
        Set noise of synthesis model to be random for every
        input.
        """
        self.G_synthesis.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        return self.G_synthesis.static_noise(
            trainable=trainable, noise_tensors=noise_tensors
        )

    def __len__(self):
        """
        Get the number of affine (style) layers of the synthesis model.
        """
        return len(self.G_synthesis)

    def truncate(self, dlatents):
        """
        Truncate the dlatents.
        Arguments:
            dlatents (torch.Tensor)
        Returns:
            truncated_dlatents (torch.Tensor)
        """
        if self.layer_psi is not None:
            dlatents = lerp(self.dlatent_avg, dlatents, self.layer_psi)
        return dlatents

    def forward(
        self,
        latents=None,
        labels=None,
        dlatents=None,
        return_dlatents=False,
        mapping_grad=True,
        latent_to_layer_idx=None,
    ):
        """
        Synthesize some data from latent inputs. The latents
        can have an extra optional dimension, where latents
        from this dimension will be distributed to the different
        affine layers of the synthesis model. The distribution
        is a index to index mapping if the amount of latents
        is the same as the number of affine layers. Otherwise,
        latents are distributed consecutively for a random
        number of layers before the next latent is used for
        some random amount of following layers. If the size
        of this extra dimension is 1 or it does not exist,
        the same latent is passed to every affine layer.

        Latents are first mapped to disentangled latents (`dlatents`)
        and are then optionally truncated (if model is in eval mode
        and truncation options have been set.) Set up truncation by
        calling `set_truncation()`.
        Arguments:
            latents (torch.Tensor): The latent values of shape
                (batch_size, N, num_features) where N is an
                optional dimension. This argument is not required
                if `dlatents` is passed.
            labels (optional): A sequence of labels, one for
                each index in the batch dimension of the input.
            dlatents (torch.Tensor, optional): Skip the latent
                mapping model and feed these dlatents straight
                to the synthesis model. The same type of distribution
                to affine layers as is described in this function
                description is also used for dlatents.
                NOTE: Explicitly passing dlatents to this function
                    will stop them from being truncated. If required,
                    do this manually by calling the `truncate()` function
                    of this model.
            return_dlatents (bool): Return not only the synthesized
                data, but also the dlatents. The dlatents tensor
                will also have its `requires_grad` set to True
                before being passed to the synthesis model for
                use with pathlength regularization during training.
                This requires training to be enabled (`thismodel.train()`).
                Default value is False.
            mapping_grad (bool): Let gradients be calculated when passing
                latents through the latent mapping model. Should be
                set to False when only optimising the synthesiser parameters.
                Default value is True.
            latent_to_layer_idx (list, tuple, optional): A manual mapping
                of the latent vectors to the affine layers of this network.
                Each position in this sequence maps the affine layer of the
                same index to an index of the latents. The latents should
                have a shape of (batch_size, N, num_features) and this argument
                should be a list of the same length as number of affine layers
                in this model (can be found by calling len(thismodel)) with values
                in the range [0, N - 1]. Without this argument, latents are distributed
                according to this function description.
        """
        # Keep track of number of latents for each batch index.
        num_latents = 1

        # Keep track of if dlatent truncation is enabled or disabled.
        truncate = False

        if dlatents is None:
            # Calculate dlatents

            # dlatent truncation enabled as dlatents were not explicitly given
            truncate = True

            assert latents is not None, (
                'Either the `latents` ' + 'or the `dlatents` argument is required.'
            )
            if labels is not None:
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.int64)

            # If latents are passed with the layer dimension we need
            # to flatten it to shape (N, latent_size) before passing
            # it to the latent mapping model.
            if latents.dim() == 3:
                num_latents = latents.size(1)
                latents = latents.view(-1, latents.size(-1))
                # Labels need to repeated for the extra dimension of latents.
                if labels is not None:
                    labels = labels.unsqueeze(1).repeat(1, num_latents).view(-1)

            # Dont allow this operation to create a computation graph for
            # backprop unless specified. This is useful for pathreg as it
            # only regularizes the parameters of the synthesiser and not
            # to latent mapping model.
            with torch.set_grad_enabled(mapping_grad):
                dlatents = self.G_mapping(latents=latents, labels=labels)
        else:
            if dlatents.dim() == 3:
                num_latents = dlatents.size(1)

        # Now we expand/repeat the number of latents per batch index until it is
        # the same number as affine layers in our synthesis model.
        dlatents = dlatents.view(-1, num_latents, dlatents.size(-1))
        if num_latents == 1:
            dlatents = dlatents.expand(dlatents.size(0), len(self), dlatents.size(2))
        elif num_latents != len(self):
            assert dlatents.size(1) <= len(
                self
            ), 'More latents ({}) than number '.format(
                dlatents.size(1)
            ) + 'of generator layers ({}) received.'.format(
                len(self)
            )
            if not latent_to_layer_idx:
                # Lets randomly distribute the latents to
                # ranges of layers (each latent is assigned
                # to a random number of consecutive layers).
                cutoffs = np.random.choice(
                    np.arange(1, len(self)), dlatents.size(1) - 1, replace=False
                )
                cutoffs = [0] + sorted(cutoffs.tolist()) + [len(self)]
                dlatents = [
                    dlatents[:, i]
                    .unsqueeze(1)
                    .expand(-1, cutoffs[i + 1] - cutoffs[i], dlatents.size(2))
                    for i in range(dlatents.size(1))
                ]
                dlatents = torch.cat(dlatents, dim=1)
            else:
                # Assign latents as specified by argument
                assert len(latent_to_layer_idx) == len(self), (
                    'The latent index to layer index mapping does '
                    + 'not have the same number of elements '
                    + '({}) as the number of '.format(len(latent_to_layer_idx))
                    + 'generator layers ({})'.format(len(self))
                )
                dlatents = dlatents[:, latent_to_layer_idx]

        # Update moving average of dlatents when training
        if self.training and self.dlatent_avg_beta != 1:
            with torch.no_grad():
                batch_dlatent_avg = dlatents[:, 0].mean(dim=0)
                self.dlatent_avg = lerp(
                    batch_dlatent_avg, self.dlatent_avg, self.dlatent_avg_beta
                )

        # Truncation is only applied when dlatents are not explicitly
        # given and the model is in evaluation mode.
        if truncate and not self.training:
            dlatents = self.truncate(dlatents)

        # One of the reasons we might want to return the dlatents is for
        # pathreg, in which case the dlatents need to require gradients
        # before being passed to the synthesiser. This should only be
        # the case when the model is in training mode.
        if return_dlatents and self.training:
            dlatents.requires_grad_(True)

        synth = self.G_synthesis(latents=dlatents)
        if return_dlatents:
            return synth, dlatents
        return synth


# Base class for the parameterized models. This is used as parent
# class to reduce duplicate code and documentation for shared arguments.
class _BaseParameterizedModel(_BaseModel):
    """
    activation (str, callable, nn.Module): The non-linear
        activation function to use.
        Default value is leaky relu with a slope of 0.2.
    lr_mul (float): The learning rate multiplier for this
        model. When loading weights of previously trained
        networks, this value has to be the same as when
        the network was trained for the outputs to not
        change (as this is used to scale the weights).
        Default value depends on model type and can
        be found in the original paper for StyleGAN.
    weight_scale (bool): Use weight scaling for
        equalized learning rate. Default value
        is True.
    eps (float): Epsilon value added for numerical stability.
        Default value is 1e-8."""

    def __init__(self, **kwargs):
        super(_BaseParameterizedModel, self).__init__()
        self._update_default_kwargs(
            activation='lrelu:0.2', lr_mul=1, weight_scale=True, eps=1e-8
        )
        self._update_kwargs(**kwargs)


class GeneratorMapping(_BaseParameterizedModel):
    """
    Latent mapping model, handles the
    transformation of latents into disentangled
    latents.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. Unused by default.
        out_size (int, optional): The size of the disentangled
            latents output by this model. If not specified,
            the outputs will have the same size as the input
            latents.
        num_layers (int): Number of dense layers in this
            model. Default value is 8.
        hidden (int, optional): Number of hidden features of layers.
            If unspecified, this is the same size as the latents.
        normalize_input (bool): Normalize the input of this
            model. Default value is True."""

    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorMapping, self).__init__()
        self._update_default_kwargs(
            latent_size=512,
            label_size=0,
            out_size=None,
            num_layers=8,
            hidden=None,
            normalize_input=True,
            lr_mul=0.01,
        )
        self._update_kwargs(**kwargs)

        # Find in and out features of first dense layer
        in_features = self.latent_size
        out_features = self.hidden or self.latent_size

        # Each class label has its own embedded vector representation.
        self.embedding = None
        if self.label_size:
            self.embedding = nn.Embedding(self.label_size, self.latent_size)
            # The input is now the latents concatenated with
            # the label embeddings.
            in_features += self.latent_size
        dense_layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                # Set out features for last dense layer
                out_features = self.out_size or self.latent_size
            dense_layers.append(
                BiasActivationWrapper(
                    layer=DenseLayer(
                        in_features=in_features,
                        out_features=out_features,
                        lr_mul=self.lr_mul,
                        weight_scale=self.weight_scale,
                        gain=1,
                    ),
                    features=out_features,
                    use_bias=True,
                    activation=self.activation,
                    bias_init=0,
                    lr_mul=self.lr_mul,
                    weight_scale=self.weight_scale,
                )
            )
            in_features = out_features
        self.main = nn.Sequential(*dense_layers)

    def forward(self, latents, labels=None):
        """
        Get the disentangled latents from the input latents
        and optional labels.
        Arguments:
            latents (torch.Tensor): Tensor of shape (batch_size, latent_size).
            labels (torch.Tensor, optional): Labels for conditioning of latents
                if there are any.
        Returns:
            dlatents (torch.Tensor): Disentangled latents of same shape as
                `latents` argument.
        """
        assert latents.dim() == 2 and latents.size(-1) == self.latent_size, (
            'Incorrect input shape. Should be '
            + '(batch_size, {}) '.format(self.latent_size)
            + 'but received {}'.format(tuple(latents.size()))
        )
        x = latents
        if labels is not None:
            assert self.embedding is not None, (
                'No embedding layer found, please '
                + 'specify the number of possible labels '
                + 'in the constructor of this class if '
                + 'using labels.'
            )
            assert len(labels) == len(latents), (
                'Received different number of labels '
                + '({}) and latents ({}).'.format(len(labels), len(latents))
            )
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.int64)
            assert labels.dtype == torch.int64, (
                'Labels should be integer values ' + 'of dtype torch.in64 (long)'
            )
            y = self.embedding(labels)
            x = torch.cat([x, y], dim=-1)
        else:
            assert self.embedding is None, 'Missing input labels.'
        if self.normalize_input:
            x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.main(x)


# Base class for the synthesising and discriminating models. This is used as parent
# class to reduce duplicate code and documentation for shared arguments.
class _BaseAdverserialModel(_BaseParameterizedModel):
    """
    data_channels (int): Number of channels of the data.
        Default value is 3.
    base_shape (list, tuple): This is the shape of the feature
        activations when it is most compact and still has the
        same number of dims as the data. This is one of the
        arguments that controls what shape the data will be.
        the value of each size in the shape is going to double
        in size for number of `channels` - 1.
        Example:
            `data_channels=3`
            `base_shape=(4, 2)`
            and 9 `channels` in total will give us a shape of
            (3, 4 * 2^(9 - 1), 2 * 2^(9 - 1)) which is the
            same as (3, 1024, 512).
        Default value is (4, 4).
    channels (int, list, optional): The channels of each block
        of layers. If int, this many channel values will be
        created with sensible default values optimal for image
        synthesis. If list, the number of blocks in this model
        will be the same as the number of channels in the list.
        Default value is the int value 9 which will create the
        following channels: [32, 32, 64, 128, 256, 512, 512, 512, 512].
        These are the channel values used in the stylegan2 paper for
        their FFHQ-trained face generation network.
        If channels is given as a list it should be in the order:
            Generator: last layer -> first layer
            Discriminator: first layer -> last layer
    resnet (bool): Use resnet connections.
        Defaults:
            Generator: False
            Discriminator: True
    skip (bool): Use skip connections for data.
        Defaults:
            Generator: True
            Discriminator: False
    fused_resample (bool): Fuse any up- or downsampling that
        is paired with a convolutional layer into a strided
        convolution (transposed if upsampling was used).
        Default value is True.
    conv_resample_mode (str): The resample mode of up- or
        downsampling layers. If `fused_resample=True` only
        'FIR' and 'none' can be used. Else, 'FIR' or anything
        that can be passed to torch.nn.functional.interpolate
        is a valid mode (and 'max' but only for downsampling
        operations). Default value is 'FIR'.
    conv_filter (int, list): The filter to use if
        `conv_resample_mode='FIR'`. If int, a low
        pass filter of this size will be used. If list,
        the filter is explicitly specified. If the filter
        is of a single dimension it will be expanded to
        the number of dimensions of the data. Default
        value is a low pass filter of [1, 3, 3, 1].
    skip_resample_mode (str): If `skip=True`, this
        mode is used for the resamplings of skip
        connections of different sizes. Same possible
        values as `conv_filter` (except 'none', which
        can not be used). Default value is 'FIR'.
    skip_filter (int, list): Same description as
        `conv_filter` but for skip connections.
        Only used if `skip_resample_mode='FIR'` and
        `skip=True`. Default value is a low pass
        filter of [1, 3, 3, 1].
    kernel_size (int): The size of the convolutional kernels.
        Default value is 3.
    conv_pad_mode (str): The padding mode for convolutional
        layers. Has to be one of 'constant', 'reflect',
        'replicate' or 'circular'. Default value is
        'constant'.
    conv_pad_constant (float): The value to use for conv
        padding if `conv_pad_mode='constant'`. Default
        value is 0.
    filter_pad_mode (str): The padding mode for FIR
        filters. Same possible values as `conv_pad_mode`.
        Default value is 'constant'.
    filter_pad_constant (float): The value to use for FIR
        padding if `filter_pad_mode='constant'`. Default
        value is 0.
    pad_once (bool): If FIR filter is used in conjunction with a
        conv layer, do all the padding for both convolution and
        FIR in the FIR layer instead of once per layer.
        Default value is True.
    conv_block_size (int): The number of conv layers in
        each conv block. Default value is 2."""

    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(_BaseAdverserialModel, self).__init__()
        self._update_default_kwargs(
            data_channels=3,
            base_shape=(4, 4),
            channels=9,
            resnet=False,
            skip=False,
            fused_resample=True,
            conv_resample_mode='FIR',
            conv_filter=[1, 3, 3, 1],
            skip_resample_mode='FIR',
            skip_filter=[1, 3, 3, 1],
            kernel_size=3,
            conv_pad_mode='constant',
            conv_pad_constant=0,
            filter_pad_mode='constant',
            filter_pad_constant=0,
            pad_once=True,
            conv_block_size=2,
        )
        self._update_kwargs(**kwargs)

        self.dim = len(self.base_shape)
        assert 1 <= self.dim <= 3, '`base_shape` can only have 1, 2 or 3 dimensions.'
        if isinstance(self.channels, int):
            # Create the specified number of channel values with sensible
            # sizes (these values do well for image synthesis).
            num_channels = self.channels
            self.channels = [min(32 * 2 ** i, 512) for i in range(min(8, num_channels))]
            if len(self.channels) < num_channels:
                self.channels = [32] * (
                    num_channels - len(self.channels)
                ) + self.channels


class GeneratorSynthesis(_BaseAdverserialModel):
    """
    The synthesis model that takes latents and synthesises
    some data.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        modulate_data_out (bool): Apply style to the data output
            layers. These layers are projections of the feature
            maps into the space of the data. Default value is True.
        noise (bool): Add noise after each conv style layer.
            Default value is True."""

    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorSynthesis, self).__init__()
        self._update_default_kwargs(
            latent_size=512,
            demodulate=True,
            modulate_data_out=True,
            noise=True,
            resnet=False,
            skip=True,
        )
        self._update_kwargs(**kwargs)

        # The constant input of the model has no activations
        # normalization, it is just passed straight to the first
        # layer of the model.
        self.const = torch.nn.Parameter(
            torch.empty(self.channels[-1], *self.base_shape).normal_()
        )
        conv_block_kwargs = dict(
            latent_size=self.latent_size,
            demodulate=self.demodulate,
            resnet=self.resnet,
            up=True,
            num_layers=self.conv_block_size,
            filter=self.conv_filter,
            activation=self.activation,
            mode=self.conv_resample_mode,
            fused=self.fused_resample,
            kernel_size=self.kernel_size,
            pad_mode=self.conv_pad_mode,
            pad_constant=self.conv_pad_constant,
            filter_pad_mode=self.filter_pad_mode,
            filter_pad_constant=self.filter_pad_constant,
            pad_once=self.pad_once,
            noise=self.noise,
            lr_mul=self.lr_mul,
            weight_scale=self.weight_scale,
            gain=1,
            dim=self.dim,
            eps=self.eps,
        )
        self.conv_blocks = nn.ModuleList()

        # The first convolutional layer is slightly different
        # from the following convolutional blocks but can still
        # be represented as a convolutional block if we change
        # some of its arguments.
        self.conv_blocks.append(
            GeneratorConvBlock(
                **{
                    **conv_block_kwargs,
                    'in_channels': self.channels[-1],
                    'out_channels': self.channels[-1],
                    'resnet': False,
                    'up': False,
                    'num_layers': 1,
                }
            )
        )

        # The rest of the convolutional blocks all look the same
        # except for number of input and output channels
        for i in range(1, len(self.channels)):
            self.conv_blocks.append(
                GeneratorConvBlock(
                    in_channels=self.channels[-i],
                    out_channels=self.channels[-i - 1],
                    **conv_block_kwargs,
                )
            )

        # If not using the skip architecture, only one
        # layer will project the feature maps into
        # the space of the data (from the activations of
        # the last convolutional block). If using the skip
        # architecture, every block will have its
        # own projection layer instead.
        self.to_data_layers = nn.ModuleList()
        for i in range(1, len(self.channels) + 1):
            to_data = None
            if i == len(self.channels) or self.skip:
                to_data = BiasActivationWrapper(
                    layer=ConvLayer(
                        **{
                            **conv_block_kwargs,
                            'in_channels': self.channels[-i],
                            'out_channels': self.data_channels,
                            'modulate': self.modulate_data_out,
                            'demodulate': False,
                            'kernel_size': 1,
                        }
                    ),
                    **{
                        **conv_block_kwargs,
                        'features': self.data_channels,
                        'use_bias': True,
                        'activation': 'linear',
                        'bias_init': 0,
                    },
                )
            self.to_data_layers.append(to_data)

        # When the skip architecture is used we need to
        # upsample data outputs of previous convolutional
        # blocks so that it can be added to the data output
        # of the current convolutional block.
        self.upsample = None
        if self.skip:
            self.upsample = Upsample(
                mode=self.skip_resample_mode,
                filter=self.skip_filter,
                filter_pad_mode=self.filter_pad_mode,
                filter_pad_constant=self.filter_pad_constant,
                gain=1,
                dim=self.dim,
            )

        # Calculate the number of latents required
        # in the input.
        self._num_latents = 1 + self.conv_block_size * (len(self.channels) - 1)
        # Only the final data output layer uses
        # its own latent input when being modulated.
        # The other data output layers recycles latents
        # from the next convolutional block.
        if self.modulate_data_out:
            self._num_latents += 1

    def __len__(self):
        """
        Get the number of affine (style) layers of this model.
        """
        return self._num_latents

    def random_noise(self):
        """
        Set injected noise to be random for each new input.
        """
        for module in self.modules():
            if isinstance(module, NoiseInjectionWrapper):
                module.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        rtn_tensors = []

        if not self.noise:
            return rtn_tensors

        for module in self.modules():
            if isinstance(module, NoiseInjectionWrapper):
                has_noise_shape = module.has_noise_shape()
                device = module.weight.device
                dtype = module.weight.dtype
                break

        # If noise layers dont have the shape that the noise should be
        # we first need to pass some data through the network once for
        # these layers to record the shape. To create noise tensors
        # we need to know what size they should be.
        if not has_noise_shape:
            with torch.no_grad():
                self(
                    torch.zeros(
                        1, len(self), self.latent_size, device=device, dtype=dtype
                    )
                )

        i = 0
        for block in self.conv_blocks:
            for layer in block.conv_block:
                for module in layer.modules():
                    if isinstance(module, NoiseInjectionWrapper):
                        noise_tensor = None
                        if noise_tensors is not None:
                            if i < len(noise_tensors):
                                noise_tensor = noise_tensors[i]
                                i += 1
                            else:
                                rtn_tensors.append(None)
                                continue
                        rtn_tensors.append(
                            module.static_noise(
                                trainable=trainable, noise_tensor=noise_tensor
                            )
                        )

        if noise_tensors is not None:
            assert len(rtn_tensors) == len(noise_tensors), (
                'Got a list of {} '.format(len(noise_tensors))
                + 'noise tensors but there are '
                + '{} noise layers in this model'.format(len(rtn_tensors))
            )

        return rtn_tensors

    def forward(self, latents):
        """
        Synthesise some data from input latents.
        Arguments:
            latents (torch.Tensor): Latent vectors of shape
                (batch_size, num_affine_layers, latent_size)
                where num_affine_layers is the value returned
                by __len__() of this class.
        Returns:
            synthesised (torch.Tensor): Synthesised data.
        """
        assert latents.dim() == 3 and latents.size(1) == len(self), (
            'Input mismatch, expected latents of shape '
            + '(batch_size, {}, latent_size) '.format(len(self))
            + 'but got {}.'.format(tuple(latents.size()))
        )
        # Declare our feature activations variable
        # and give it the value of our const parameter with
        # an added batch dimension.
        x = self.const.unsqueeze(0)
        # Declare our data (output) variable
        y = None
        # Start counting style layers used. This is used for specifying
        # which latents should be passed to the current block in the loop.
        layer_idx = 0
        for block, to_data in zip(self.conv_blocks, self.to_data_layers):
            # Get the latents for the style layers in this block.
            block_latents = latents[:, layer_idx : layer_idx + len(block)]

            x = block(input=x, latents=block_latents)

            layer_idx += len(block)

            # Upsample the data output of the previous block to fit
            # the data output size of this block so that they can
            # be added together. Only performed for 'skip' architectures.
            if self.upsample is not None and layer_idx < len(self):
                if y is not None:
                    y = self.upsample(y)

            # Combine the data output of this block with any previous
            # blocks outputs if using 'skip' architecture, else only
            # perform this operation for the very last block outputs.
            if to_data is not None:
                t = to_data(input=x, latent=latents[:, layer_idx])
                y = t if y is None else y + t
        return y


class Discriminator(_BaseAdverserialModel):
    """
    The discriminator scores data inputs.
    Keyword Arguments:
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. The discriminator will calculate scores
            for each possible label and only returns the score
            from the label passed with the input data. If no
            labels are used, only one score is calculated.
            Disabled by default.
        mbstd_group_size (int): Group size for minibatch std
            before the final conv layer. A value of 0 indicates
            not to use minibatch std, and a value of -1 indicates
            that the group should be over the entire batch.
            This is used for increasing variety of the outputs of
            the generator. Default value is 4.
            NOTE: Scores for the same data may vary depending
                on batch size when using a value of -1.
            NOTE: If a value > 0 is given, every input batch
                must have a size evenly divisible by this value.
        dense_hidden (int, optional): The number of hidden features
            of the first dense layer. By default, this is the same as
            the number of channels in the final conv layer."""

    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self._update_default_kwargs(
            label_size=0, mbstd_group_size=4, dense_hidden=None, resnet=True, skip=False
        )
        self._update_kwargs(**kwargs)

        conv_block_kwargs = dict(
            resnet=self.resnet,
            down=True,
            num_layers=self.conv_block_size,
            filter=self.conv_filter,
            activation=self.activation,
            mode=self.conv_resample_mode,
            fused=self.fused_resample,
            kernel_size=self.kernel_size,
            pad_mode=self.conv_pad_mode,
            pad_constant=self.conv_pad_constant,
            filter_pad_mode=self.filter_pad_mode,
            filter_pad_constant=self.filter_pad_constant,
            pad_once=self.pad_once,
            noise=False,
            lr_mul=self.lr_mul,
            weight_scale=self.weight_scale,
            gain=1,
            dim=self.dim,
            eps=self.eps,
        )
        self.conv_blocks = nn.ModuleList()

        # All but the last of the convolutional blocks look the same
        # except for number of input and output channels
        for i in range(len(self.channels) - 1):
            self.conv_blocks.append(
                DiscriminatorConvBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    **conv_block_kwargs,
                )
            )

        # The final convolutional layer is slightly different
        # from the previous convolutional blocks but can still
        # be represented as a convolutional block if we change
        # some of its arguments and optionally add a minibatch
        # std layer before it.
        final_conv_block = []
        if self.mbstd_group_size:
            final_conv_block.append(
                MinibatchStd(group_size=self.mbstd_group_size, eps=self.eps)
            )
        final_conv_block.append(
            DiscriminatorConvBlock(
                **{
                    **conv_block_kwargs,
                    'in_channels': self.channels[-1]
                    + (1 if self.mbstd_group_size else 0),
                    'out_channels': self.channels[-1],
                    'resnet': False,
                    'down': False,
                    'num_layers': 1,
                },
            )
        )
        self.conv_blocks.append(nn.Sequential(*final_conv_block))

        # If not using the skip architecture, only one
        # layer will project the data into feature maps.
        # This would be performed only for the input data at
        # the first block.
        # If using the skip architecture, every block will
        # have its own projection layer instead.
        self.from_data_layers = nn.ModuleList()
        for i in range(len(self.channels)):
            from_data = None
            if i == 0 or self.skip:
                from_data = BiasActivationWrapper(
                    layer=ConvLayer(
                        **{
                            **conv_block_kwargs,
                            'in_channels': self.data_channels,
                            'out_channels': self.channels[i],
                            'modulate': False,
                            'demodulate': False,
                            'kernel_size': 1,
                        }
                    ),
                    **{
                        **conv_block_kwargs,
                        'features': self.channels[i],
                        'use_bias': True,
                        'activation': self.activation,
                        'bias_init': 0,
                    },
                )
            self.from_data_layers.append(from_data)

        # When the skip architecture is used we need to
        # downsample the data input so that it has the same
        # size as the feature maps of each block so that it
        # can be projected and added to these feature maps.
        self.downsample = None
        if self.skip:
            self.downsample = Downsample(
                mode=self.skip_resample_mode,
                filter=self.skip_filter,
                filter_pad_mode=self.filter_pad_mode,
                filter_pad_constant=self.filter_pad_constant,
                gain=1,
                dim=self.dim,
            )

        # The final layers are two dense layers that maps
        # the features into score logits. If labels are
        # used, we instead output one score for each possible
        # class of the labels and then return the score for the
        # labeled class.
        dense_layers = []
        in_features = self.channels[-1] * np.prod(self.base_shape)
        out_features = self.dense_hidden or self.channels[-1]
        activation = self.activation
        for _ in range(2):
            dense_layers.append(
                BiasActivationWrapper(
                    layer=DenseLayer(
                        in_features=in_features,
                        out_features=out_features,
                        lr_mul=self.lr_mul,
                        weight_scale=self.weight_scale,
                        gain=1,
                    ),
                    features=out_features,
                    activation=activation,
                    use_bias=True,
                    bias_init=0,
                    lr_mul=self.lr_mul,
                    weight_scale=self.weight_scale,
                )
            )
            in_features = out_features
            out_features = max(1, self.label_size)
            activation = 'linear'
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, input, labels=None):
        """
        Takes some data and optionally its labels and
        produces one score logit per data input.
        Arguments:
            input (torch.Tensor)
            labels (torch.Tensor, list, optional)
        Returns:
            score_logits (torch.Tensor)
        """
        # Declare our feature activations variable.
        x = None
        # Declare our data (input) variable
        y = input
        for i, (block, from_data) in enumerate(
            zip(self.conv_blocks, self.from_data_layers)
        ):
            # Combine the data input of this block with any previous
            # block output if using 'skip' architecture, else only
            # perform this operation as a way to create inputs for
            # the first block.
            if from_data is not None:
                t = from_data(y)
                x = t if x is None else x + t

            x = block(input=x)

            # Downsample the data input of this block to fit
            # the feature size of the output of this block so that they can
            # be added together. Only performed for 'skip' architectures.
            if self.downsample is not None and i != len(self.conv_blocks) - 1:
                y = self.downsample(y)
        # Calculate scores
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        if labels is not None:
            # Use advanced indexing to fetch only the score of the
            # class labels.
            x = x[torch.arange(x.size(0)), labels].unsqueeze(-1)
        return x


def get_activation(activation):
    """
    Get the module for a specific activation function and its gain if
    it can be calculated.
    Arguments:
        activation (str, callable, nn.Module): String representing the activation.
    Returns:
        activation_module (torch.nn.Module): The module representing
            the activation function.
        gain (float): The gain value. Defaults to 1 if it can not be calculated.
    """
    if isinstance(activation, nn.Module) or callable(activation):
        return activation, 1.0
    if isinstance(activation, str):
        activation = activation.lower()
    if activation in [None, 'linear']:
        return nn.Identity(), 1.0
    lrelu_strings = ('leaky', 'leakyrely', 'leaky_relu', 'leaky relu', 'lrelu')
    if activation.startswith(lrelu_strings):
        for l_s in lrelu_strings:
            activation = activation.replace(l_s, '')
        slope = ''.join(char for char in activation if char.isdigit() or char == '.')
        slope = float(slope) if slope else 0.01
        return nn.LeakyReLU(slope), np.sqrt(2)  # close enough to true gain
    elif activation.startswith('swish'):
        return Swish(affine=activation != 'swish'), np.sqrt(2)
    elif activation in ['relu']:
        return nn.ReLU(), np.sqrt(2)
    elif activation in ['elu']:
        return nn.ELU(), 1.0
    elif activation in ['prelu']:
        return nn.PReLU(), np.sqrt(2)
    elif activation in ['rrelu', 'randomrelu']:
        return nn.RReLU(), np.sqrt(2)
    elif activation in ['selu']:
        return nn.SELU(), 1.0
    elif activation in ['softplus']:
        return nn.Softplus(), 1
    elif activation in ['softsign']:
        return nn.Softsign(), 1  # unsure about this gain
    elif activation in ['sigmoid', 'logistic']:
        return nn.Sigmoid(), 1.0
    elif activation in ['tanh']:
        return nn.Tanh(), 1.0
    else:
        raise ValueError('Activation "{}" not available.'.format(activation))


class Swish(nn.Module):
    """
    Performs the 'Swish' non-linear activation function.
    https://arxiv.org/pdf/1710.05941.pdf
    Arguments:
        affine (bool): Multiply the input to sigmoid
            with a learnable scale. Default value is False.
    """

    def __init__(self, affine=False):
        super(Swish, self).__init__()
        if affine:
            self.beta = nn.Parameter(torch.tensor([1.0]))
        self.affine = affine

    def forward(self, input, *args, **kwargs):
        """
        Apply the swish non-linear activation function
        and return the results.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        if self.affine:
            x *= self.beta
        return x * torch.sigmoid(x)


def _get_weight_and_coef(shape, lr_mul=1, weight_scale=True, gain=1, fill=None):
    """
    Get an intialized weight and its runtime coefficients as an nn.Parameter tensor.
    Arguments:
        shape (tuple, list): Shape of weight tensor.
        lr_mul (float): The learning rate multiplier for
            this weight. Default value is 1.
        weight_scale (bool): Use weight scaling for equalized
            learning rate. Default value is True.
        gain (float): The gain of the weight. Default value is 1.
        fill (float, optional): Instead of initializing the weight
            with scaled normally distributed values, fill it with
            this value. Useful for bias weights.
    Returns:
        weight (nn.Parameter)
    """
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)

    if weight_scale:
        init_std = 1 / lr_mul
        runtime_coef = he_std * lr_mul
    else:
        init_std = he_std / lr_mul
        runtime_coef = lr_mul

    weight = torch.empty(*shape)
    if fill is None:
        weight.normal_(0, init_std)
    else:
        weight.fill_(fill)
    return nn.Parameter(weight), runtime_coef


def _apply_conv(input, *args, transpose=False, **kwargs):
    """
    Perform a 1d, 2d or 3d convolution with specified
    positional and keyword arguments. Which type of
    convolution that is used depends on shape of data.
    Arguments:
        input (torch.Tensor): The input data for the
            convolution.
        *args: Positional arguments for the convolution.
    Keyword Arguments:
        transpose (bool): Transpose the convolution.
            Default value is False
        **kwargs: Keyword arguments for the convolution.
    """
    dim = input.dim() - 2
    conv_fn = getattr(F, 'conv{}{}d'.format('_transpose' if transpose else '', dim))
    return conv_fn(input=input, *args, **kwargs)


def _setup_mod_weight_for_t_conv(weight, in_channels, out_channels):
    """
    Reshape a modulated conv weight for use with a transposed convolution.
    Arguments:
        weight (torch.Tensor)
        in_channels (int)
        out_channels (int)
    Returns:
        reshaped_weight (torch.Tensor)
    """
    # [BO]I*k -> BOI*k
    weight = weight.view(-1, out_channels, in_channels, *weight.size()[2:])
    # BOI*k -> BIO*k
    weight = weight.transpose(1, 2)
    # BIO*k -> [BI]O*k
    weight = weight.reshape(-1, out_channels, *weight.size()[3:])
    return weight


def _setup_filter_kernel(filter_kernel, gain=1, up_factor=1, dim=2):
    """
    Set up a filter kernel and return it as a tensor.
    Arguments:
        filter_kernel (int, list, torch.tensor, None): The filter kernel
            values to use. If this value is an int, a binomial filter of
            this size is created. If a sequence with a single axis is used,
            it will be expanded to the number of `dims` specified. If value
            is None, a filter of values [1, 1] is used.
        gain (float): Gain of the filter kernel. Default value is 1.
        up_factor (int): Scale factor. Should only be given for upscaling filters.
            Default value is 1.
        dim (int): Number of dimensions of data. Default value is 2.
    Returns:
        filter_kernel_tensor (torch.Tensor)
    """
    filter_kernel = filter_kernel or 2
    if isinstance(filter_kernel, (int, float)):

        def binomial(n, k):
            if k in [1, n]:
                return 1
            return np.math.factorial(n) / (
                np.math.factorial(k) * np.math.factorial(n - k)
            )

        filter_kernel = [
            binomial(filter_kernel, k) for k in range(1, filter_kernel + 1)
        ]
    if not torch.is_tensor(filter_kernel):
        filter_kernel = torch.tensor(filter_kernel)
    filter_kernel = filter_kernel.float()
    if filter_kernel.dim() == 1:
        _filter_kernel = filter_kernel.unsqueeze(0)
        while filter_kernel.dim() < dim:
            filter_kernel = torch.matmul(filter_kernel.unsqueeze(-1), _filter_kernel)
    assert all(filter_kernel.size(0) == s for s in filter_kernel.size())
    filter_kernel /= filter_kernel.sum()
    filter_kernel *= gain * up_factor ** 2
    return filter_kernel.float()


def _get_layer(layer_class, kwargs, wrap=False, noise=False):
    """
    Create a layer and wrap it in optional
    noise and/or bias/activation layers.
    Arguments:
        layer_class: The class of the layer to construct.
        kwargs (dict): The keyword arguments to use for constructing
            the layer and optionally the bias/activaiton layer.
        wrap (bool): Wrap the layer in an bias/activation layer and
            optionally a noise injection layer. Default value is False.
        noise (bool): Inject noise before the bias/activation wrapper.
            This can only be done when `wrap=True`. Default value is False.
    """
    layer = layer_class(**kwargs)
    if wrap:
        if noise:
            layer = NoiseInjectionWrapper(layer)
        layer = BiasActivationWrapper(layer, **kwargs)
    return layer


class BiasActivationWrapper(nn.Module):
    """
    Wrap a module to add bias and non-linear activation
    to any outputs of that module.
    Arguments:
        layer (nn.Module): The module to wrap.
        features (int, optional): The number of features
            of the output of the `layer`. This argument
            has to be specified if `use_bias=True`.
        use_bias (bool): Add bias to the output.
            Default value is True.
        activation (str, nn.Module, callable, optional):
            non-linear activation function to use.
            Unused if notspecified.
        bias_init (float): Value to initialize bias
            weight with. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the bias weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
    """

    def __init__(
        self,
        layer,
        features=None,
        use_bias=True,
        activation='linear',
        bias_init=0,
        lr_mul=1,
        weight_scale=True,
        *args,
        **kwargs
    ):
        super(BiasActivationWrapper, self).__init__()
        self.layer = layer
        bias = None
        bias_coef = None
        if use_bias:
            assert features, '`features` is required when using bias.'
            bias, bias_coef = _get_weight_and_coef(
                shape=[features], lr_mul=lr_mul, weight_scale=False, fill=bias_init
            )
        self.register_parameter('bias', bias)
        self.bias_coef = bias_coef
        self.act, self.gain = get_activation(activation)

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        bias (if set) and run through non-linear activation
        function (if set).
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        if self.bias is not None:
            bias = self.bias.view(1, -1, *[1] * (x.dim() - 2))
            if self.bias_coef != 1:
                bias = self.bias_coef * bias
            x += bias
        x = self.act(x)
        if self.gain != 1:
            x *= self.gain
        return x

    def extra_repr(self):
        return 'bias={}'.format(self.bias is not None)


class NoiseInjectionWrapper(nn.Module):
    """
    Wrap a module to add noise scaled by a
    learnable parameter to any outputs of the
    wrapped module.
    Noise is randomized for each output but can
    be set to static noise by calling `static_noise()`
    of this object. This can only be done once data
    has passed through this layer at least once so that
    the shape of the static noise to create is known.
    Check if the shape is known by calling `has_noise_shape()`.
    Arguments:
        layer (nn.Module): The module to wrap.
        same_over_batch (bool): Repeat the same
            noise values over the entire batch
            instead of creating separate noise
            values for each entry in the batch.
            Default value is True.
    """

    def __init__(self, layer, same_over_batch=True):
        super(NoiseInjectionWrapper, self).__init__()
        self.layer = layer
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('noise_storage', None)
        self.same_over_batch = same_over_batch
        self.random_noise()

    def has_noise_shape(self):
        """
        If this module has had data passed through it
        the noise shape is known and this function returns
        True. Else False.
        Returns:
            noise_shape_known (bool)
        """
        return self.noise_storage is not None

    def random_noise(self):
        """
        Randomize noise for each
        new output.
        """
        self._fixed_noise = False
        if isinstance(self.noise_storage, nn.Parameter):
            noise_storage = self.noise_storage
            del self.noise_storage
            self.register_buffer('noise_storage', noise_storage.data)

    def static_noise(self, trainable=False, noise_tensor=None):
        """
        Set up static noise that can optionally be a trainable
        parameter. Static noise does not change between inputs
        unless the user has altered its values. Returns the tensor
        object that stores the static noise.
        Arguments:
            trainable (bool): Wrap the static noise tensor in
                nn.Parameter to make it trainable. The returned
                tensor will be wrapped.
            noise_tensor (torch.Tensor, optional): A predefined
                static noise tensor. If not passed, one will be
                created.
        """
        assert self.has_noise_shape(), 'Noise shape is unknown'
        if noise_tensor is None:
            noise_tensor = self.noise_storage
        else:
            noise_tensor = noise_tensor.to(self.weight)
        if trainable and not isinstance(noise_tensor, nn.Parameter):
            noise_tensor = nn.Parameter(noise_tensor)
        if isinstance(self.noise_storage, nn.Parameter) and not trainable:
            del self.noise_storage
            self.register_buffer('noise_storage', noise_tensor)
        else:
            self.noise_storage = noise_tensor
        self._fixed_noise = True
        return noise_tensor

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        submodule in :meth:`~torch.nn.Module.state_dict`.

        Overridden to ignore the noise storage buffer.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if name != 'noise_storage' and param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if name != 'noise_storage' and buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Overridden to ignore noise storage buffer.
        """
        key = prefix + 'noise_storage'
        if key in state_dict:
            del state_dict[key]
        return super(NoiseInjectionWrapper, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        noise to its outputs before returning them.
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        noise_shape = list(x.size())
        noise_shape[1] = 1
        if self.same_over_batch:
            noise_shape[0] = 1
        if self.noise_storage is None or list(self.noise_storage.size()) != noise_shape:
            if not self._fixed_noise:
                self.noise_storage = torch.empty(
                    *noise_shape, dtype=self.weight.dtype, device=self.weight.device
                )
            else:
                assert list(self.noise_storage.size()[2:]) == noise_shape[2:], (
                    'A data size {} has been encountered, '.format(x.size()[2:])
                    + 'the static noise previously set up does '
                    + 'not match this size {}'.format(self.noise_storage.size()[2:])
                )
                assert self.noise_storage.size(0) == 1 or self.noise_storage.size(
                    0
                ) == x.size(0), (
                    'Static noise batch size mismatch! '
                    + 'Noise batch size: {}, '.format(self.noise_storage.size(0))
                    + 'input batch size: {}'.format(x.size(0))
                )
                assert self.noise_storage.size(1) == 1 or self.noise_storage.size(
                    1
                ) == x.size(1), (
                    'Static noise channel size mismatch! '
                    + 'Noise channel size: {}, '.format(self.noise_storage.size(1))
                    + 'input channel size: {}'.format(x.size(1))
                )
        if not self._fixed_noise:
            self.noise_storage.normal_()
        x += self.weight * self.noise_storage
        return x

    def extra_repr(self):
        return 'static_noise={}'.format(self._fixed_noise)


class FilterLayer(nn.Module):
    """
    Apply a filter by using convolution.
    Arguments:
        filter_kernel (torch.Tensor): The filter kernel to use.
            Should be of shape `dims * (k,)` where `k` is the
            kernel size and `dims` is the number of data dimensions
            (excluding batch and channel dimension).
        stride (int): The stride of the convolution.
        pad0 (int): Amount to pad start of each data dimension.
            Default value is 0.
        pad1 (int): Amount to pad end of each data dimension.
            Default value is 0.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
    """

    def __init__(
        self,
        filter_kernel,
        stride=1,
        pad0=0,
        pad1=0,
        pad_mode='constant',
        pad_constant=0,
        *args,
        **kwargs
    ):
        super(FilterLayer, self).__init__()
        dim = filter_kernel.dim()
        filter_kernel = filter_kernel.view(1, 1, *filter_kernel.size())
        self.register_buffer('filter_kernel', filter_kernel)
        self.stride = stride
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
            self.pad_mode = pad_mode
            self.pad_constant = pad_constant

    def forward(self, input, **kwargs):
        """
        Pad the input and run the filter over it
        before returning the new values.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        conv_kwargs = dict(
            weight=self.filter_kernel.repeat(
                input.size(1), *[1] * (self.filter_kernel.dim() - 1)
            ),
            stride=self.stride,
            groups=input.size(1),
        )
        if self.fused_pad:
            conv_kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, transpose=False, **conv_kwargs)

    def extra_repr(self):
        return 'filter_size={}, stride={}'.format(
            tuple(self.filter_kernel.size()[2:]), self.stride
        )


class Upsample(nn.Module):
    """
    Performs upsampling without learnable parameters that doubles
    the size of data.
    Arguments:
        mode (str): 'FIR' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(
        self,
        mode='FIR',
        filter=[1, 3, 3, 1],
        filter_pad_mode='constant',
        filter_pad_constant=0,
        gain=1,
        dim=2,
        *args,
        **kwargs
    ):
        super(Upsample, self).__init__()
        assert mode != 'max', 'mode \'max\' can only be used for downsampling.'
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter, gain=gain, up_factor=2, dim=dim
            )
            pad = filter_kernel.size(-1) - 1
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                pad0=(pad + 1) // 2 + 1,
                pad1=pad // 2,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant,
            )
            self.register_buffer('weight', torch.ones(*[1 for _ in range(dim + 2)]))
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Upsample inputs.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = _apply_conv(
                input=input,
                weight=self.weight.expand(input.size(1), *self.weight.size()[1:]),
                groups=input.size(1),
                stride=2,
                transpose=True,
            )
            x = self.filter(x)
        else:
            interp_kwargs = dict(scale_factor=2, mode=self.mode)
            if 'linear' in self.mode or 'cubic' in self.mode:
                interp_kwargs.update(align_corners=False)
            x = F.interpolate(input, **interp_kwargs)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class Downsample(nn.Module):
    """
    Performs downsampling without learnable parameters that
    reduces size of data by half.
    Arguments:
        mode (str): 'FIR', 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(
        self,
        mode='FIR',
        filter=[1, 3, 3, 1],
        filter_pad_mode='constant',
        filter_pad_constant=0,
        gain=1,
        dim=2,
        *args,
        **kwargs
    ):
        super(Downsample, self).__init__()
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter, gain=gain, up_factor=1, dim=dim
            )
            pad = filter_kernel.size(-1) - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                stride=2,
                pad0=pad0,
                pad1=pad1,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant,
            )
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Downsample inputs to half its size.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = self.filter(input)
        elif self.mode == 'max':
            return getattr(F, 'max_pool{}d'.format(input.dim() - 2))(input)
        else:
            x = F.interpolate(input, scale_factor=0.5, mode=self.mode)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class MinibatchStd(nn.Module):
    """
    Adds the aveage std of each data point over a
    slice of the minibatch to that slice as a new
    feature map. This gives an output with one extra
    channel.
    Arguments:
        group_size (int): Number of entries in each slice
            of the batch. If <= 0, the entire batch is used.
            Default value is 4.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(self, group_size=4, eps=1e-8, *args, **kwargs):
        super(MinibatchStd, self).__init__()
        if group_size is None or group_size <= 0:
            # Entire batch as group size
            group_size = 0
        assert group_size != 1, 'Can not use 1 as minibatch std group size.'
        self.group_size = group_size
        self.eps = eps

    def forward(self, input, **kwargs):
        """
        Add a new feature map to the input containing the average
        standard deviation for each slice.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        group_size = self.group_size or input.size(0)
        assert input.size(0) >= group_size, (
            'Can not use a smaller batch size '
            + '({}) than the specified '.format(input.size(0))
            + 'group size ({}) '.format(group_size)
            + 'of this minibatch std layer.'
        )
        assert input.size(0) % group_size == 0, (
            'Can not use a batch of a size '
            + '({}) that is not '.format(input.size(0))
            + 'evenly divisible by the group size ({})'.format(group_size)
        )
        x = input

        # B = batch size, C = num channels
        # *s = the size dimensions (height, width for images)

        # BC*s -> G[B/G]C*s
        y = input.view(group_size, -1, *input.size()[1:])
        # For numerical stability when training with mixed precision
        y = y.float()
        # G[B/G]C*s
        y -= y.mean(dim=0, keepdim=True)
        # [B/G]C*s
        y = torch.mean(y ** 2, dim=0)
        # [B/G]C*s
        y = torch.sqrt(y + self.eps)
        # [B/G]
        y = torch.mean(y.view(y.size(0), -1), dim=-1)
        # [B/G]1*1
        y = y.view(-1, *[1] * (input.dim() - 1))
        # Cast back to input dtype
        y = y.to(x)
        # B1*1
        y = y.repeat(group_size, *[1] * (y.dim() - 1))
        # B1*s
        y = y.expand(y.size(0), 1, *x.size()[2:])
        # B[C+1]*s
        x = torch.cat([x, y], dim=1)
        return x

    def extra_repr(self):
        return 'group_size={}'.format(self.group_size or '-1')


class DenseLayer(nn.Module):
    """
    A fully connected layer.
    NOTE: No bias is applied in this layer.
    Arguments:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
    """

    def __init__(
        self,
        in_features,
        out_features,
        lr_mul=1,
        weight_scale=True,
        gain=1,
        *args,
        **kwargs
    ):
        super(DenseLayer, self).__init__()
        weight, weight_coef = _get_weight_and_coef(
            shape=[out_features, in_features],
            lr_mul=lr_mul,
            weight_scale=weight_scale,
            gain=gain,
        )
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef

    def forward(self, input, **kwargs):
        """
        Perform a matrix multiplication of the weight
        of this layer and the input.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        return input.matmul(weight.t())

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.weight.size(1), self.weight.size(0)
        )


class ConvLayer(nn.Module):
    """
    A convolutional layer that can have its outputs
    modulated (style mod). It can also normalize outputs.
    These operations are done by modifying the convolutional
    kernel weight and employing grouped convolutions for
    efficiency.
    NOTE: No bias is applied in this layer.
    NOTE: Amount of padding used is the same as 'SAME'
        argument in tensorflow for conv padding.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int, optional): The size of the
            latents to use for modulating this convolution.
            Only required when `modulate=True`.
        modulate (bool): Applies a "style" to the outputs
            of the layer. The style is given by a latent
            vector passed with the input to this layer.
            A dense layer is added that projects the
            values of the latent into scales for the
            data channels.
            Default value is False.
        demodulate (bool): Normalize std of outputs.
            Can only be set to True when `modulate=True`.
            Default value is False.
        kernel_size (int): The size of the kernel.
            Default value is 3.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_size=None,
        modulate=False,
        demodulate=False,
        kernel_size=3,
        pad_mode='constant',
        pad_constant=0,
        lr_mul=1,
        weight_scale=True,
        gain=1,
        dim=2,
        eps=1e-8,
        *args,
        **kwargs
    ):
        super(ConvLayer, self).__init__()
        assert modulate or not demodulate, (
            '`demodulate=True` can ' + 'only be used when `modulate=True`'
        )
        if modulate:
            assert latent_size is not None, (
                'When using `modulate=True`, ' + '`latent_size` has to be specified.'
            )
        kernel_shape = [out_channels, in_channels] + dim * [kernel_size]
        weight, weight_coef = _get_weight_and_coef(
            shape=kernel_shape, lr_mul=lr_mul, weight_scale=weight_scale, gain=gain
        )
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef
        if modulate:
            self.dense = BiasActivationWrapper(
                layer=DenseLayer(
                    in_features=latent_size,
                    out_features=in_channels,
                    lr_mul=lr_mul,
                    weight_scale=weight_scale,
                    gain=1,
                ),
                features=in_channels,
                use_bias=True,
                activation='linear',
                bias_init=1,
                lr_mul=lr_mul,
                weight_scale=weight_scale,
            )
        self.dense_reshape = [-1, 1, in_channels] + dim * [1]
        self.dmod_reshape = [-1, out_channels, 1] + dim * [1]
        pad = kernel_size - 1
        pad0 = pad - pad // 2
        pad1 = pad - pad0
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
        self.pad_mode = pad_mode
        self.pad_constant = pad_constant
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.modulate = modulate
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.lr_mul = lr_mul
        self.weight_scale = weight_scale
        self.gain = gain
        self.dim = dim
        self.eps = eps

    def forward_mod(self, input, latent, weight, **kwargs):
        """
        Run the forward operation with modulation.
        Automatically called from `forward()` if modulation
        is enabled.
        """
        assert latent is not None, (
            'A latent vector is '
            + 'required for the forwad pass of a modulated conv layer.'
        )

        # B = batch size, C = num channels
        # *s = the size dimensions, example: (height, width) for images
        # *k = sizes of the convolutional kernel excluding in and out channel dimensions.
        # *1 = multiple dimensions of size 1, with number of dimensions depending on data format.
        # O = num output channels, I = num input channels

        # BI
        style_mod = self.dense(input=latent)
        # B1I*1
        style_mod = style_mod.view(*self.dense_reshape)
        # 1OI*k
        weight = weight.unsqueeze(0)
        # (1OI*k)x(B1I*1) -> BOI*k
        weight = weight * style_mod
        if self.demodulate:
            # BO
            dmod = torch.rsqrt(
                torch.sum(weight.view(weight.size(0), weight.size(1), -1) ** 2, dim=-1)
                + self.eps
            )
            # BO1*1
            dmod = dmod.view(*self.dmod_reshape)
            # (BOI*k)x(BO1*1) -> BOI*k
            weight = weight * dmod
        # BI*s -> 1[BI]*s
        x = input.view(1, -1, *input.size()[2:])
        # BOI*k -> [BO]I*k
        weight = weight.view(-1, *weight.size()[2:])
        # 1[BO]*s
        x = self._process(input=x, weight=weight, groups=input.size(0))
        # 1[BO]*s -> BO*s
        x = x.view(-1, self.out_channels, *x.size()[2:])
        return x

    def forward(self, input, latent=None, **kwargs):
        """
        Convolve the input.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor, optional)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        if self.modulate:
            return self.forward_mod(input=input, latent=latent, weight=weight)
        return self._process(input=input, weight=weight)

    def _process(self, input, weight, **kwargs):
        """
        Pad input and convolve it returning the result.
        """
        x = input
        if self.fused_pad:
            kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, weight=weight, transpose=False, **kwargs)

    def extra_repr(self):
        string = 'in_channels={}, out_channels={}'.format(
            self.weight.size(1), self.weight.size(0)
        )
        string += ', modulate={}, demodulate={}'.format(self.modulate, self.demodulate)
        return string


class ConvUpLayer(ConvLayer):
    """
    A convolutional upsampling layer that doubles the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the upsampling operation with the
            convolution, turning this layer into a strided transposed
            convolution. Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(
        self,
        *args,
        fused=True,
        mode='FIR',
        filter=[1, 3, 3, 1],
        filter_pad_mode='constant',
        filter_pad_constant=0,
        pad_once=True,
        **kwargs
    ):
        super(ConvUpLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], (
                'Fused conv upsample can only use '
                + '\'FIR\' or \'none\' for resampling '
                + '(`mode` argument).'
            )
            self.padding = np.ceil(self.kernel_size / 2 - 1)
            self.output_padding = 2 * (self.padding + 1) - self.kernel_size
            if not self.modulate:
                # pre-prepare weights only once instead of every forward pass
                self.weight = nn.Parameter(
                    self.weight.data.transpose(0, 1).contiguous()
                )
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(
                    filter_kernel=filter, gain=self.gain, up_factor=2, dim=self.dim
                )
                if pad_once:
                    self.padding = 0
                    self.output_padding = 0
                    pad = (filter_kernel.size(-1) - 2) - (self.kernel_size - 1)
                    pad0 = ((pad + 1) // 2 + 1,)
                    pad1 = (pad // 2 + 1,)
                else:
                    pad = filter_kernel.size(-1) - 1
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant,
                )
        else:
            assert mode != 'none', (
                '\'none\' can not be used as '
                + 'sampling `mode` when `fused=False` as upsampling '
                + 'has to be performed separately from the conv layer.'
            )
            self.upsample = Upsample(
                mode=mode,
                filter=filter,
                filter_pad_mode=filter_pad_mode,
                filter_pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim,
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            if self.modulate:
                weight = _setup_mod_weight_for_t_conv(
                    weight=weight,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                )
            pad_out = False
            if self.pad_mode == 'constant' and self.pad_constant == 0:
                if self.filter is not None or not self.pad_once:
                    kwargs.update(
                        padding=self.padding,
                        output_padding=self.output_padding,
                    )
            elif self.filter is None:
                if self.padding:
                    x = F.pad(
                        x,
                        [self.padding] * 2 * self.dim,
                        mode=self.pad_mode,
                        value=self.pad_constant,
                    )
                pad_out = self.output_padding != 0
            kwargs.update(stride=2)
            x = _apply_conv(input=x, weight=weight, transpose=True, **kwargs)
            if pad_out:
                x = F.pad(
                    x,
                    [self.output_padding, 0] * self.dim,
                    mode=self.pad_mode,
                    value=self.pad_constant,
                )
            if self.filter is not None:
                x = self.filter(x)
        else:
            x = super(ConvUpLayer, self)._process(
                input=self.upsample(input=x), weight=weight, **kwargs
            )
        return x

    def extra_repr(self):
        string = super(ConvUpLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


class ConvDownLayer(ConvLayer):
    """
    A convolutional downsampling layer that halves the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the downsampling operation with the
            convolution, turning this layer into a strided convolution.
            Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(
        self,
        *args,
        fused=True,
        mode='FIR',
        filter=[1, 3, 3, 1],
        filter_pad_mode='constant',
        filter_pad_constant=0,
        pad_once=True,
        **kwargs
    ):
        super(ConvDownLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], (
                'Fused conv downsample can only use '
                + '\'FIR\' or \'none\' for resampling '
                + '(`mode` argument).'
            )
            pad = self.kernel_size - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            if pad0 == pad1 and (
                pad0 == 0 or self.pad_mode == 'constant' and self.pad_constant == 0
            ):
                self.fused_pad = True
                self.padding = pad0
            else:
                self.fused_pad = False
                self.padding = [pad0, pad1] * self.dim
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(
                    filter_kernel=filter, gain=self.gain, up_factor=1, dim=self.dim
                )
                if pad_once:
                    self.fused_pad = True
                    self.padding = 0
                    pad = (filter_kernel.size(-1) - 2) + (self.kernel_size - 1)
                    pad0 = ((pad + 1) // 2,)
                    pad1 = (pad // 2,)
                else:
                    pad = filter_kernel.size(-1) - 1
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant,
                )
                self.pad_once = pad_once
        else:
            assert mode != 'none', (
                '\'none\' can not be used as '
                + 'sampling `mode` when `fused=False` as downsampling '
                + 'has to be performed separately from the conv layer.'
            )
            self.downsample = Downsample(
                mode=mode,
                filter=filter,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim,
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            kwargs.update(stride=2)
            if self.filter is not None:
                x = self.filter(input=x)
        else:
            x = self.downsample(input=x)
        x = super(ConvDownLayer, self)._process(input=x, weight=weight, **kwargs)
        return x

    def extra_repr(self):
        string = super(ConvDownLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(self.fused, self.mode)
        return string


class GeneratorConvBlock(nn.Module):
    """
    A convblock for the synthesiser model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int): The size of the latent vectors.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        up (bool): Upsample the data to twice its size. This is
            performed in the first layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `up=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of upsampling layers.
            Only used when `up=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `up=True`, fuse the upsample operation
            and the first convolutional layer into a transposed
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        noise (bool): Add noise to the output of each layer. Default value
            is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_size,
        demodulate=True,
        resnet=False,
        up=False,
        num_layers=2,
        filter=[1, 3, 3, 1],
        activation='leaky:0.2',
        mode='FIR',
        fused=True,
        kernel_size=3,
        pad_mode='constant',
        pad_constant=0,
        filter_pad_mode='constant',
        filter_pad_constant=0,
        pad_once=True,
        use_bias=True,
        noise=True,
        lr_mul=1,
        weight_scale=True,
        gain=1,
        dim=2,
        eps=1e-8,
        *args,
        **kwargs
    ):
        super(GeneratorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(
            features=out_channels,
            modulate=True,
        )

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if up:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, (
                '`mode` {} '.format(mode)
                + 'is not one of the available sample '
                + 'modes {}.'.format(available_sampling)
            )

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            use_up = up and not self.conv_block
            self.conv_block.append(
                _get_layer(
                    ConvUpLayer if use_up else ConvLayer,
                    layer_kwargs,
                    wrap=True,
                    noise=noise,
                )
            )
            layer_kwargs.update(in_channels=out_channels)

        self.projection = None
        if resnet:
            projection_kwargs = {
                **layer_kwargs,
                'in_channels': in_channels,
                'kernel_size': 1,
                'modulate': False,
                'demodulate': False,
            }
            self.projection = _get_layer(
                ConvUpLayer if up else ConvLayer, projection_kwargs, wrap=False
            )

        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, latents=None, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if latents.dim() == 2:
            latents.unsqueeze(1)
        if latents.size(1) == 1:
            latents = latents.repeat(1, len(self), 1)
        assert latents.size(1) == len(self), (
            'Number of latent inputs '
            + '({}) does not match '.format(latents.size(1))
            + 'number of conv layers '
            + '({}) in block.'.format(len(self))
        )
        x = input
        for i, layer in enumerate(self.conv_block):
            x = layer(input=x, latent=latents[:, i])
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x


class DiscriminatorConvBlock(nn.Module):
    """
    A convblock for the discriminator model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        down (bool): Downsample the data to twice its size. This is
            performed in the last layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `down=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of downsampling layers.
            Only used when `down=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, 'max' or anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `down=True`, fuse the downsample operation
            and the last convolutional layer into a strided
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resnet=False,
        down=False,
        num_layers=2,
        filter=[1, 3, 3, 1],
        activation='leaky:0.2',
        mode='FIR',
        fused=True,
        kernel_size=3,
        pad_mode='constant',
        pad_constant=0,
        filter_pad_mode='constant',
        filter_pad_constant=0,
        pad_once=True,
        use_bias=True,
        lr_mul=1,
        weight_scale=True,
        gain=1,
        dim=2,
        *args,
        **kwargs
    ):
        super(DiscriminatorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(
            out_channels=in_channels,
            features=in_channels,
            modulate=False,
            demodulate=False,
        )

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if down:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('max')
                available_sampling.append('area')
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, (
                '`mode` {} '.format(mode)
                + 'is not one of the available sample '
                + 'modes {}'.format(available_sampling)
            )

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            if len(self.conv_block) == num_layers - 1:
                layer_kwargs.update(out_channels=out_channels, features=out_channels)
            use_down = down and len(self.conv_block) == num_layers - 1
            self.conv_block.append(
                _get_layer(
                    ConvDownLayer if use_down else ConvLayer,
                    layer_kwargs,
                    wrap=True,
                    noise=False,
                )
            )

        self.projection = None
        if resnet:
            projection_kwargs = {
                **layer_kwargs,
                'in_channels': in_channels,
                'kernel_size': 1,
                'modulate': False,
                'demodulate': False,
            }
            self.projection = _get_layer(
                ConvDownLayer if down else ConvLayer, projection_kwargs, wrap=False
            )

        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        for layer in self.conv_block:
            x = layer(input=x)
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x


class MovingAverageModule:
    def __init__(
        self, from_module, to_module=None, param_beta=0.995, buffer_beta=0, device=None
    ):
        from_module = unwrap_module(from_module)
        to_module = unwrap_module(to_module)
        if device is None:
            module = from_module
            if to_module is not None:
                module = to_module
            device = next(module.parameters()).device
        else:
            device = torch.device(device)
        self.from_module = from_module
        if to_module is None:
            self.module = from_module.clone().to(device)
        else:
            assert type(to_module) == type(
                from_module
            ), 'Mismatch between type of source and target module.'
            assert set(self._get_named_parameters(to_module).keys()) == set(
                self._get_named_parameters(from_module).keys()
            ), 'Mismatch between parameters of source and target module.'
            assert set(self._get_named_buffers(to_module).keys()) == set(
                self._get_named_buffers(from_module).keys()
            ), 'Mismatch between buffers of source and target module.'
            self.module = to_module.to(device)
        self.module.eval().requires_grad_(False)
        self.param_beta = param_beta
        self.buffer_beta = buffer_beta
        self.device = device

    def __getattr__(self, name):
        try:
            return super(object, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def update(self):
        self._update_data(
            from_data=self._get_named_parameters(self.from_module),
            to_data=self._get_named_parameters(self.module),
            beta=self.param_beta,
        )
        self._update_data(
            from_data=self._get_named_buffers(self.from_module),
            to_data=self._get_named_buffers(self.module),
            beta=self.buffer_beta,
        )

    @staticmethod
    def _update_data(from_data, to_data, beta):
        for name in from_data.keys():
            if name not in to_data:
                continue
            fr, to = from_data[name], to_data[name]
            with torch.no_grad():
                if beta == 0:
                    to.data.copy_(fr.data.to(to.data))
                elif beta < 1:
                    to.data.copy_(lerp(fr.data.to(to.data), to.data, beta))

    @staticmethod
    def _get_named_parameters(module):
        return {name: value for name, value in module.named_parameters()}

    @staticmethod
    def _get_named_buffers(module):
        return {name: value for name, value in module.named_buffers()}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.module.eval()
        args, args_in_device = move_to_device(args, self.device)
        kwargs, kwargs_in_device = move_to_device(kwargs, self.device)
        in_device = None
        if args_in_device is not None:
            in_device = args_in_device
        if kwargs_in_device is not None:
            in_device = kwargs_in_device
        out = self.module(*args, **kwargs)
        if in_device is not None:
            out, _ = move_to_device(out, in_device)
        return out


_WRAPPER_CLASSES = (
    MovingAverageModule,
    nn.DataParallel,
    nn.parallel.DistributedDataParallel,
)


def unwrap_module(module):
    if isinstance(module, _WRAPPER_CLASSES):
        return module.module
    return module


def lerp(a, b, beta):
    if isinstance(beta, numbers.Number):
        if beta == 1:
            return b
        elif beta == 0:
            return a
    if torch.is_tensor(a) and a.dtype == torch.float32:
        # torch lerp only available for fp32
        return torch.lerp(a, b, beta)
    # More numerically stable than a + beta * (b - a)
    return (1 - beta) * a + beta * b


def move_to_device(value, device):
    if torch.is_tensor(value):
        value.to(device), value.device
    orig_device = None
    if isinstance(value, (tuple, list)):
        values = []
        for val in value:
            _val, orig_device = move_to_device(val, device)
            values.append(_val)
        return type(value)(values), orig_device
    if isinstance(value, dict):
        if isinstance(value, collections.OrderedDict):
            values = collections.OrderedDict()
        else:
            values = {}
        for key, val in value.items():
            _val, orig_device = move_to_device(val, device)
            values[key] = val
        return values, orig_device
    return value, orig_device
