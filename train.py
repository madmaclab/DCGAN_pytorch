from __future__ import division

import argparse
import torch
import torch.nn as nn
import model
import loader
from torch.autograd import Variable
import tensorflow as tf
from StringIO import StringIO
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import numpy as np


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def maybe_makedir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def main(args):
    lr = float(args.lr)
    n_epoch = args.epoch
    batch_size = args.batch_size

    writer = tf.summary.FileWriter(args.save_dir + '/tfsummary')

    dataset = loader.dataset(args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    D, G = model.model(type=args.model_type)
    loss = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    for epoch in range(n_epoch):
        for i, (real_image, _) in enumerate(data_loader):
            real_images = to_var(real_image.view(batch_size, -1))  # only when we use DNN, not CNN

            # real label
            output_real = D(real_images)
            d_loss_real = loss(output_real, to_var(torch.ones(batch_size)))

            # fake label
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            output_fake = D(fake_images)
            d_loss_fake = loss(output_fake, to_var(torch.zeros(batch_size)))

            # train D
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # train G
            z = to_var(torch.randn(batch_size, 64))
            fake_images = G(z)
            output_fake_2 = D(fake_images)

            # is it maximize log(D(G(z)) ??
            g_loss = loss(output_fake_2, to_var(torch.ones(batch_size)))
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if (i+1) % 300 == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, D(real): %.2f, D(G(z)): %.2f'
                      %(epoch, n_epoch, i+1, len(dataset)//batch_size, d_loss.data[0], g_loss.data[0],
                        d_loss_real.data.mean(), d_loss_fake.data.mean()))
                summaries = list()
                summaries.append(tf.Summary.Value(tag='d_loss_real',
                                                  simple_value=d_loss_real.data.mean()))
                summaries.append(tf.Summary.Value(tag='d_loss_fake',
                                                  simple_value=d_loss_fake.data.mean()))
                summaries.append(tf.Summary.Value(tag='g_loss',
                                                  simple_value=g_loss))
                sample_images = to_np(loader.denorm(fake_images.view(-1, 28, 28)[:5]))

                img = np.concatenate(sample_images, axis=1)
                s = StringIO()
                plt.imsave(s, img, format='png', vmin=0, vmax=1, cmap='gray')
                img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=img.shape[0],
                                           width=img.shape[1])
                summaries.append(tf.Summary.Value(tag='generated_image',
                                                  image=img_sum))
                summary = tf.Summary(value=summaries)
                writer.add_summary(summary, global_step=epoch*len(dataset)//batch_size + i)
        if (epoch + 1) == 1:
            images = real_images.view(batch_size, 1, 28, 28)
            save_image(loader.denorm(images.data), args.save_dir + '/real_images.png')

        # Save sampled images
        fake_images = fake_images.view(batch_size, 1, 28, 28)
        save_image(loader.denorm(fake_images.data), args.save_dir + '/fake_images-%d.png' % (epoch + 1))

    # Save the trained parameters
    torch.save(G.state_dict(), args.save_dir + '/generator.pkl')
    torch.save(D.state_dict(), args.save_dir + './discriminator.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset: mnist / f-mnist")
    parser.add_argument("save_dir", type=str, help="save_dir")
    parser.add_argument("--lr", type=str, default='3e-4', help="learning rate")
    parser.add_argument("--epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="n_batch")
    parser.add_argument("--model_type", type=str, default='basic', help="model_type")
    args = parser.parse_args()
    maybe_makedir(args.save_dir)

    main(args)