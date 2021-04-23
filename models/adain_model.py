'''
https://github.com/naoto0804/pytorch-AdaIN/blob/master/net.py
'''
import torch.nn as nn

from util.functions import adaptive_instance_normalization as adain


class Generator(nn.Module):
    def __init__(self, encoder, decoder, lambda_rec=1.2, lambda_fm=1):
        """
        λfm = 1 for all experiments, λrec = 1.2 for the face2anime dataset
        λrec = 2 for the selfie2anime dataset
        :param encoder:
        :param decoder:
        :param lembda_rec:
        :param lambd_fm:
        """
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.lambda_rec = lambda_rec
        self.lambda_fm = lambda_fm

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def decoder(self, input):
        # ASC
        # FST*x
        return None

    def discriminator(self, input):
        # double discriminator
        # image
        dl_x = None  # realface domain
        dl_y = None  # animeface domain
        return dl_x, dl_y

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # style + content encode
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        # decoder
        self.decode(content_feat, style_feats)
        # TODO: replace with adapolin
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        # discriminator
        dl_x, dl_y = self.discriminator(input)

        # loss functions
        L_adv = adv_loss()
        L_rec = rec_loss()
        L_fm = fm_loss()
        L_dfm = dfm_loss()
        L_adv + self.lambda_rec * L_rec + self.lambda_fm * (L_fm + L_dfm)
        -L_adv
        ###end?
