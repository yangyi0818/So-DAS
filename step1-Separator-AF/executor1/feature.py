#!/user/bin/env python

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections.abc import Sequence
MATH_PI = math.pi

# https://github.com/Sanyuan-Chen/CSS_with_Conformer/blob/b221bcd111e832a15f954cc2081e9affc9bdf3ae/executor/feature.py#L394
class FeatureExtractor(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, normalize=True, round_pow_of_two=True, num_spks=2, window="sqrt_hann", log_spectrogram=True, mvn_spectrogram=True, \
                       ipd_mean_normalize=True, ipd_mean_normalize_version=2, ipd_cos=True, ipd_sin=True, \
                       ipd_index="1,0;2,0;3,0;4,0;5,0;6,0", ang_index="1,0;2,0;3,0;4,0;5,0;6,0", do_ipd=False, do_doa=False):

        super(FeatureExtractor, self).__init__()
        self.forward_stft = STFT(frame_len, frame_hop, normalize=normalize, window=window, round_pow_of_two=round_pow_of_two)
        self.inverse_stft = iSTFT(frame_len, frame_hop, normalize=normalize, round_pow_of_two=round_pow_of_two)
        self.has_spatial = False
        num_bins = self.forward_stft.num_bins
        self.feature_dim = num_bins
        self.num_bins = num_bins
        self.num_spks = num_spks
        self.mvn_mag = mvn_spectrogram
        self.log_mag = log_spectrogram

        # IPD or not
        self.do_ipd = do_ipd
        self.ipd_extractor = None
        if do_ipd:
            self.ipd_extractor = IPDFeature(ipd_index, cos=ipd_cos, sin=ipd_sin, ipd_mean_normalize_version=ipd_mean_normalize_version, ipd_mean_normalize=ipd_mean_normalize)
            self.feature_dim += self.ipd_extractor.num_pairs * num_bins
            self.has_spatial = True

        # AF or not
        self.do_doa = do_doa
        self.ang_extractor = None
        if do_doa:
            self.ang_extractor = AngleFeature(num_bins=num_bins, num_doas=1, af_index=ang_index) # must known the DoA
            self.feature_dim += num_bins * self.num_spks
            self.has_spatial = True

    def stft(self, x, cplx=False):
        return self.forward_stft(x, cplx=cplx)

    def istft(self, m, p, cplx=False):
        return self.inverse_stft(m, p, cplx=cplx)

    def compute_spectra(self, x):
        """
        Function: Compute spectra features
        Input:  x:         [b c n]   (multi-channel) or [b 1 n] (single channel)
        Output: mag & pha: [b c f t] (multi-channel) or [b f t] (single channel)
                feature:   [b * t]
        """
        mag, pha = self.forward_stft(x)
        # ch0: b x f x t
        if mag.dim() == 4:
            f = th.clamp(mag[:, 0], min=1e-8)
        else:
            f = th.clamp(mag, min=1e-8)
        # log
        if self.log_mag:
            f = th.log(f)
        # mvn
        if self.mvn_mag:
            f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-8)

        return mag, pha, f


    def compute_spatial(self, x, doa=None, pha=None):
        """
        Function: Compute spatial features
        Input:  pha:     [b c f t]
        Output: feature: [b * t]
        """
        if pha is None:
             _, pha = self.forward_stft(x) # [b c f t]
        else:
            if pha.dim() != 4:
                raise RuntimeError("Expect phase matrix a 4D tensor, " + f"got {pha.dim()} instead")

        feature = []
        if self.has_spatial:
            if self.do_ipd:
                ipd = self.ipd_extractor(pha)
                #print('ipd:', ipd.shape)
                feature.append(ipd) # [b mf t]
            if self.do_doa:
                ang = self.ang_extractor(pha, doa)
                #print('ang:', ang.shape)
                feature.append(ang) # [b sf t]

        feature = th.cat(feature, 1)
        return feature


    def forward(self, x, doa=None):
        """
        Input:  x:         [b c n]   (multi-channel) or [b 1 n] (single channel)
        Output: mag & pha: [b c f t] (multi-channel) or [b f t] (single channel)
                feature:   [b * t]
        """
        mag, pha, f = self.compute_spectra(x)
        feature = [f]
        if self.has_spatial:
            spatial = self.compute_spatial(x, pha=pha, doa=doa)
            feature.append(spatial)
        # b x * x t
        feature = th.cat(feature, 1)

        return mag, pha, feature


# https://github.com/Sanyuan-Chen/CSS_with_Conformer/blob/b221bcd111e832a15f954cc2081e9affc9bdf3ae/executor/feature.py#L19
def init_kernel(frame_len,
                frame_hop,
                normalize=True,
                round_pow_of_two=True,
                window="sqrt_hann"):
    if window != "sqrt_hann" and window != "hann":
        raise RuntimeError("Now only support sqrt hanning window or hann window")
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = th.hann_window(frame_len)
    if window == "sqrt_hann":
        W = W**0.5
    # scale factor to make same magnitude after iSTFT
    if window == "sqrt_hann" and normalize:
        S = 0.5 * (N * N / frame_hop)**0.5
    else:
        S = 1
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 normalize=True,
                 round_pow_of_two=True):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2
        if window == "hann":
            self.conjugate = True
        else:
            self.conjugate = False

    def extra_repr(self):
        return (f"window={self.window}, stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
            if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                i = -i
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.contiguous().view(N * C, 1, S)
            # NC x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
            if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                i = -i
        if cplx:
            return r, i
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s


# https://github.com/Sanyuan-Chen/CSS_with_EETransformer/blob/master/executor/feature.py#L170
class IPDFeature(nn.Module):
    def __init__(self, ipd_index="1,0;2,0;3,0;4,0;5,0;6,0", cos=True, sin=False, ipd_mean_normalize_version=2, ipd_mean_normalize=True):

        super(IPDFeature, self).__init__()
        split_index = lambda sstr: [tuple(map(int, p.split(","))) for p in sstr.split(";")]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.ipd_mean_normalize=ipd_mean_normalize
        self.ipd_mean_normalize_version=ipd_mean_normalize_version
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p):
        """
        Function: Accept multi-channel phase and output inter-channel phase difference
        Input:  phase matrix [b c f t]
        Output: ipd          [b mf t]
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.ipd_mean_normalize:
            yr = th.cos(pha_dif)
            yi = th.sin(pha_dif)
            yrm = yr.mean(-1, keepdim=True)
            yim = yi.mean(-1, keepdim=True)
            if self.ipd_mean_normalize_version == 1:
                pha_dif = th.atan2(yi - yim, yr - yrm)
            elif self.ipd_mean_normalize_version == 2:
                pha_dif_mean = th.atan2(yim, yrm)
                pha_dif -= pha_dif_mean
            elif self.ipd_mean_normalize_version == 3:
                pha_dif_mean = pha_dif.mean(-1, keepdim=True)
                pha_dif -= pha_dif_mean
            else:
                # we only support version 1, 2 and 3
                raise RuntimeError(
                    "{} expect ipd_mean_normalization version 1 or version 2, but got {:d} instead".format(
                        self.__name__, self.ipd_mean_normalize_version))

        if self.cos:
            # N x M x F x T
            ipd = th.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T, along frequency axis
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            # th.fmod behaves differently from np.mod for the input that is less than -math.pi
            # i believe it is a bug
            # so we need to ensure it is larger than -math.pi by adding an extra 6 * math.pi
            #ipd = th.fmod(pha_dif + math.pi, 2 * math.pi) - math.pi
            ipd = pha_dif
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd


# https://github.com/Sanyuan-Chen/CSS_with_Conformer/blob/master/executor/feature.py#L252
class AngleFeature(nn.Module):
    """
    Compute angle/directional feature
        1) num_doas == 1: we known the DoA of the target speaker
        2) num_doas != 1: we do not have that prior, so we sampled #num_doas DoAs 
                          and compute on each directions    
    """
    def __init__(self, geometric="princeton", sr=16000, velocity=340, num_bins=257, num_doas=1, af_index="1,0;2,0;3,0;4,0;5,0;6,0"):

        super(AngleFeature, self).__init__()
        if geometric not in ["princeton"]:
            raise RuntimeError("Unsupported array geometric: {}".format(geometric))
        self.geometric = geometric
        self.sr = sr
        self.num_bins = num_bins
        self.num_doas = num_doas
        self.velocity = velocity
        split_index = lambda sstr: [tuple(map(int, p.split(","))) for p in sstr.split(";")]
        # ipd index
        pair = split_index(af_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.af_index = af_index
        omega = th.tensor([math.pi * sr * f / (num_bins - 1) for f in range(num_bins)])
        # 1 x F
        self.omega = nn.Parameter(omega[None, :], requires_grad=False)

    def _oracle_phase_delay(self, doa):
        """
        Compute oracle phase delay given DoA
        args
            doa: N
        return
            phi: N x C x F or N x D x C x F
        """
        device = doa.device
        if self.num_doas != 1:
            # doa is a unused, fake parameter
            N = doa.shape[0]
            # N x D
            doa = th.linspace(0, MATH_PI * 2, self.num_doas + 1,
                              device=device)[:-1].repeat(N, 1)
        # for princeton
        # M = 7, R = 0.0425, treat M_0 as (0, 0)
        #      *3    *2
        #
        #   *4    *0    *1
        #
        #      *5    *6
        if self.geometric == "princeton":
            R = 0.0425
            zero = th.zeros_like(doa)
            # N x 7 or N x D x 7
            tau = R * th.stack([
                zero, -th.cos(doa), -th.cos(MATH_PI / 3 - doa),
                -th.cos(2 * MATH_PI / 3 - doa),
                th.cos(doa),
                th.cos(MATH_PI / 3 - doa),
                th.cos(2 * MATH_PI / 3 - doa)
            ],
                               dim=-1) / self.velocity
            # (Nx7x1) x (1xF) => Nx7xF or (NxDx7x1) x (1xF) => NxDx7xF
            phi = th.matmul(tau.unsqueeze(-1), -self.omega)
            return phi
        else:
            return None

    def extra_repr(self):
        return (
            f"geometric={self.geometric}, af_index={self.af_index}, " +
            f"sr={self.sr}, num_bins={self.num_bins}, velocity={self.velocity}, "
            + f"known_doa={self.num_doas == 1}")

    def _compute_af(self, ipd, doa):
        """
        Function: Compute angle feature
        Input:  ipd: [b c f t]  doa: DoA of the target speaker (if we known that), b or [b d] (we do not known that, sampling D DoAs instead)
        Output: af:  [b f t] or [b d f t]
        """
        # N x C x F or N x D x C x F
        d = self._oracle_phase_delay(doa)
        d = d.unsqueeze(-1)
        if self.num_doas == 1:
            dif = d[:, self.index_l] - d[:, self.index_r]
            # N x C x F x T
            af = th.cos(ipd - dif)
            # mean or sum
            af = th.mean(af, dim=1)
        else:
            # N x D x C x F x 1
            dif = d[:, :, self.index_l] - d[:, :, self.index_r]
            # N x D x C x F x T
            af = th.cos(ipd.unsqueeze(1) - dif)
            # N x D x F x T
            af = th.mean(af, dim=2)
        return af

    def forward(self, p, doa):
        """
        Function: Accept doa of the speaker & multi-channel phase, output angle feature
        Input:  doa: DoA of target/each speaker, b or [b, ...] p: phase matrix [b c f t]
        Output: af: angle feature, [b f* t] or [b d f t] (known_doa=False)
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError("{} expect 3/4D tensor, but got {:d} instead".format(self.__name__, p.dim()))
        if p.dim() == 3:
            p = p.unsqueeze(0)
        ipd = p[:, self.index_l] - p[:, self.index_r]

        if isinstance(doa, Sequence):
            if self.num_doas != 1:
                raise RuntimeError("known_doa=False, no need to pass doa as a Sequence object")
            af = [self._compute_af(ipd, spk_doa) for spk_doa in doa]
            # [b f* t]
            af = th.cat(af, 1)
        else:
            af = self._compute_af(ipd, doa)
        return af
    
