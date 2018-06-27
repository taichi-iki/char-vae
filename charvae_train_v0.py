# coding utf-8

"""
Written_by: Taichi Iki
Created_at: 2018-06-27
Abstract:

"""

import os
import pickle
import numpy as np
import json
from PIL import Image

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.variable as V
from chainer.optimizer import GradientClipping
from chainer.optimizer import WeightDecay
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer import serializers
from chainer import initializers
import chainer.cuda as cuda


class ArgSpace(object):
    def __init__(args, dir_model=None):
        # model namespace
        args.namespace = 'charvae_train'
        
        args.dir_model = dir_model
        if args.dir_model is None:
            args.dir_model = args.make_model_name(args.namespace)

        args.dataset_train_path    = 'char_images.pklb'
        args.minibatch_size_train  = 10
        args.minibatch_size_tuning = 10
        args.dir_base_model        = None
        args.train_max_epoch       = 10000
        args.save_each             = args.train_max_epoch//10
        args.gpuid                 = 0
        args.kld_function = {ep_id:min(float(ep_id)/(0.2*args.train_max_epoch), 1.0) \
                for ep_id in range(args.train_max_epoch)}
        
        args.train_sgd_lr = 1.0
        args.train_weight_decay_rate = 10e-6
        args.train_gradient_clipping_norm = 1.0
        
        args.decode_samples = ' @１1i!おあい悪良人音楽日目森林貝木本池海二ニEFQO'
        
    def make_model_name(self, namespace):
        import sys
        import re
        bn = os.path.basename(sys.argv[0])
        m = re.search('^' + re.escape(namespace) + '_(.+)\.py$', bn)
        if m is None:
            raise Exception('The file name must start with "%s_" and end with ".py"'%(namespace))
        return 'model_' + m.group(1)
    
    def store(self):
        with open(os.path.join(self.dir_model, 'args.pklb'), 'wb') as f:
            pickle.Pickler(f, protocol=2).dump(self)


class NeuralModel(chainer.Chain):
    def __init__(self, shape_x=[64, 64], h_enc_dim=1024, z_dim=32, h_dec_dim=1024):
        self.shape_x = shape_x
        self.enc_dim = 1
        for x in self.shape_x: self.enc_dim *= x
        self.z_dim = z_dim
        self.h_enc_dim = h_enc_dim
        self.h_dec_dim = h_dec_dim

        initializer = chainer.initializers.HeNormal()
        super(NeuralModel, self).__init__(
                enc_fc_h  = L.Linear(self.enc_dim, self.h_enc_dim, initialW=initializer),
                enc_fc_mu = L.Linear(self.h_enc_dim, self.z_dim, initialW=initializer),
                enc_fc_ln_var = L.Linear(self.h_enc_dim, self.z_dim, initialW=initializer),
                dec_fc_h = L.Linear(self.z_dim, self.h_dec_dim, initialW=initializer),
                dec_fc = L.Linear(self.h_dec_dim, self.enc_dim, initialW=initializer),
            )
    
    def encode(self, x):
        h = F.tanh(self.enc_fc_h(x))
        mu = self.enc_fc_mu(h)
        ln_var = self.enc_fc_ln_var(h)
        return mu, ln_var
    
    def decode(self, z, sigmoid=True):
        h = F.tanh(self.dec_fc_h(z))
        x = self.dec_fc(h)
        x = F.reshape(x, [-1]+self.shape_x)

        x = F.sigmoid(x) if sigmoid else x
        return x

    def __call__(self, x, C=1.0, k=1):
        mu, ln_var = self.encode(x)
        mb_size = mu.data.shape[0]
        
        # reconstruction loss
        rec_loss = 0
        for l in range(k):
            z = F.gaussian(mu, ln_var)
            rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False))
        rec_loss /= (k*mb_size)

        kld_loss = gaussian_kl_divergence(mu, ln_var) / mb_size
        loss = rec_loss + C*kld_loss
        
        return loss, float(rec_loss.data), float(kld_loss.data)


def main(args):
    if not os.path.exists(args.dir_model):
        os.mkdir(args.dir_model)
    log_file_name = os.path.join(args.dir_model, 'log.txt')

    args.store()
    
    dataset_train = get_dataset(args.dataset_train_path)
    
    print('The number of datasets:')
    print('dataset_train: %d'%(len(dataset_train)))
    
    # ToDo: loading a model from dir_base_model
    
    if args.gpuid >= 0:
        cuda.get_device(args.gpuid).use()
    
    target = NeuralModel()

    if args.gpuid >= 0:
        target.to_gpu(args.gpuid)
    target_opt = chainer.optimizers.SGD(lr=args.train_sgd_lr)
    target_opt.setup(target)
    target_opt.add_hook(WeightDecay(rate=args.train_weight_decay_rate))
    target_opt.add_hook(GradientClipping(args.train_gradient_clipping_norm))
    
    def save_target(save_path):
        target.to_cpu()
        serializers.save_npz(save_path, target)
        if args.gpuid >= 0:
            target.to_gpu(args.gpuid)

    def save_charvec(save_path):
        target.to_cpu()

        charvec_dict = {}
        for v, k in dataset_train:
            mu, ln_var = target.encode(v[None,:,:])
            charvec_dict[k] = mu.data[0]
        with open(save_path, 'wb') as f:
            pickle.Pickler(f).dump(charvec_dict)

        if not(args.decode_samples is None):
            for i in range(len(args.decode_samples)):
                c = args.decode_samples[i]
                array = (255*target.decode(charvec_dict[c][None, :]).data[0]).astype('uint8')
                img_size = array.shape[0]
                array = np.broadcast_to(array[:,:,None], (img_size, img_size, 3))
                Image.fromarray(array).save(save_path + str(i) + '.bmp')
        
        if args.gpuid >= 0:
            target.to_gpu(args.gpuid)
    
    print('start training')
    for ep_id in range(args.train_max_epoch):
        np.random.shuffle(dataset_train)
        
        C = args.kld_function[ep_id]
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_kld_loss = 0
        for mb_id in range(0, len(dataset_train), args.minibatch_size_train):
            mb = dataset_train[mb_id: mb_id + args.minibatch_size_train]
            mb = [v for v, k in mb]
            x = target.xp.asarray(np.stack(mb))
            loss, rec_loss, kld_loss = target(x, C=C)
            target.zerograds()
            loss.backward()
            target_opt.update()
            loss.unchain_backward()
            epoch_loss += float(loss.data)*len(mb)
            epoch_rec_loss += rec_loss*len(mb)
            epoch_kld_loss += kld_loss*len(mb)
        epoch_loss /= len(dataset_train)
        epoch_rec_loss /= len(dataset_train)
        epoch_kld_loss /= len(dataset_train)
        record = (ep_id, C, epoch_loss, epoch_rec_loss, epoch_kld_loss)
        print('ep=%d C=%.4f loss=%.4f rec_loss=%.4f kld_loss=%.4f'%record)
        with open(log_file_name, 'a') as f:
            f.write('ep=%d C=%.4f loss=%.4f rec_loss=%.4f kld_loss=%.4f\n'%record)

        # MODEL STORING
        if ep_id % args.save_each == 0:
            save_target(os.path.join(args.dir_model, 'trained_%d.model'%(ep_id)))
            save_charvec(os.path.join(args.dir_model, 'charvec_%d.pklb'%(ep_id)))
            print('model saved.')
    
    print('training done.')
    
    save_target(os.path.join(args.dir_model, 'trained_end.model'))
    save_charvec(os.path.join(args.dir_model, 'charvec_end.pklb'))
    print('the last model saved.')


def get_dataset(pathname):
    """[(img(sizeh,sizew), key)]"""
    char_images = {}
    with open(pathname, 'rb') as f:
        char_images = pickle.Unpickler(f).load()
    
    pair_list = [(v[:,:], k) for k, v in char_images.items()]
    return pair_list


if __name__ == '__main__':
    args = ArgSpace()
    main(args)

