"""
CycleGAN PyTorch

Author: Sigve Røkenes
Date: November, 2021

Implemented as part of my CycleGAN project in Computational Creativity (TDT12) at NTNU

"""


import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision

import matplotlib.pyplot as plt

from collections import deque

from model import Generator, Discriminator
from metrics import Metrics
import data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CycleGan:

    def __init__(self, gen_a, gen_b, dsc_a, dsc_b, metrics=None):
        self.gen_a = gen_a
        self.gen_b = gen_b
        self.dsc_a = dsc_a
        self.dsc_b = dsc_b

        gen_lr = 0.0002
        dsc_lr = 0.0002

        self.gen_optim = torch.optim.Adam(
            list(self.gen_a.parameters()) + list(self.gen_b.parameters()),
            lr=gen_lr
        )
        self.dsc_a_optim = torch.optim.Adam(self.dsc_a.parameters(), lr=dsc_lr, betas=(0.5, 0.999))
        self.dsc_b_optim = torch.optim.Adam(self.dsc_b.parameters(), lr=dsc_lr, betas=(0.5, 0.999))

        self.gan_criterion = torch.nn.BCELoss()
        self.cycle_criterion = torch.nn.L1Loss()
        self.id_criterion = torch.nn.L1Loss()

        self.metrics = metrics
        self.cycle_coeff = 10.0
        self.id_coeff = 5.0

    @staticmethod
    def get_target(pred, real=True):
        if real:
            return torch.ones_like(pred, device=device)
        return torch.zeros_like(pred, device=device)

    def dsc_loss(self, dsc, real, fake, lbl=None):
        pred_real = dsc(real.clone().detach())
        pred_fake = dsc(fake.clone().detach())
        loss_real = self.gan_criterion(pred_real, self.get_target(pred_real, True))
        loss_fake = self.gan_criterion(pred_fake, self.get_target(pred_fake, False))
        if lbl is not None:
            self.metrics.record_scalar(f"dsc/pred_{lbl}_real", pred_real.mean().item())
            self.metrics.record_scalar(f"dsc/pred_{lbl}_fake", pred_fake.mean().item())
        return (loss_real + loss_fake) / 2.0

    def gen_loss(self, real_a, real_b, fake_a, fake_b, recon_a, recon_b):

        # GAN loss
        pred_a = self.dsc_a(fake_a)
        pred_b = self.dsc_b(fake_b)
        gan_loss_a = self.gan_criterion(pred_a, self.get_target(pred_a, True))
        gan_loss_b = self.gan_criterion(pred_b, self.get_target(pred_b, True))
        gan_loss = gan_loss_a + gan_loss_b

        # Identity loss
        id_a = self.id_criterion(self.gen_a(real_b), real_b) * self.id_coeff
        id_b = self.id_criterion(self.gen_b(real_a), real_a) * self.id_coeff
        id_loss = id_a + id_b

        # Cycle loss
        cycle_aba = self.cycle_criterion(recon_a, real_a) * self.cycle_coeff
        cycle_bab = self.cycle_criterion(recon_b, real_b) * self.cycle_coeff
        cycle_loss = cycle_aba + cycle_bab

        # Gen/dsc losses
        gen_loss = gan_loss + cycle_loss + id_loss

        self.metrics.record_scalar("loss/g_gan", gan_loss.item())
        self.metrics.record_scalar("loss/g_cycle", cycle_loss.item())
        self.metrics.record_scalar("loss/g_id", id_loss.item())
        self.metrics.record_scalar("loss/g_ic_coeff", self.id_coeff)
        self.metrics.record_scalar("loss/g_ic_coeff", self.cycle_coeff)

        return gen_loss

    def epoch(self, real_a, real_b):

        fake_a = self.gen_a(real_b)
        fake_b = self.gen_b(real_a)
        recon_a = self.gen_a(fake_b)
        recon_b = self.gen_b(fake_a)

        # Train generators
        gen_loss = self.gen_loss(
            real_a, real_b,
            fake_a, fake_b,
            recon_a, recon_b
        )
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        # Train discriminator A
        dsc_loss_a = self.dsc_loss(self.dsc_a, real_a, fake_a, lbl="a")
        self.dsc_a_optim.zero_grad()
        dsc_loss_a.backward()
        self.dsc_a_optim.step()

        # Train discriminator B
        dsc_loss_b = self.dsc_loss(self.dsc_b, real_b, fake_b, lbl="b")
        self.dsc_b_optim.zero_grad()
        dsc_loss_b.backward()
        self.dsc_b_optim.step()

        # Metrics
        self.metrics.record_scalar("loss/dsc_a", dsc_loss_a.item())
        self.metrics.record_scalar("loss/dsc_b", dsc_loss_b.item())


class Trainer:

    def __init__(self, batch_size=16):

        self.metrics = Metrics("CycleGan")
        print("Traing on path", self.metrics.path)

        gen_a = Generator(img_chn=3, model_chn=64, n_resblock=9).to(device)
        gen_b = Generator(img_chn=3, model_chn=64, n_resblock=9).to(device)

        dsc_a = Discriminator(
            img_size=128, img_chn=3, model_chn=64,
            n_downsample=7, n_resblock=0,
            fc_out=False, out_act=torch.nn.Sigmoid()
        ).to(device)

        dsc_b = Discriminator(
            img_size=128, img_chn=3, model_chn=64,
            n_downsample=7, n_resblock=0,
            fc_out=False, out_act=torch.nn.Sigmoid()
        ).to(device)

        self.cyclegan = CycleGan(gen_a, gen_b, dsc_a, dsc_b, metrics=self.metrics)

        print("=" * 10)
        print("Generator:\n", gen_a)
        print("=" * 10)
        print("Discriminator:\n", dsc_a)
        print("=" * 10)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1, saturation=0.1),
        ])

        self.load_a = DataLoader(
            data.get_local("cartoon", transform),
            batch_size=batch_size,
            shuffle=True
        )
        self.load_b = DataLoader(
            data.get_local("celeba_align", transform),
            batch_size=batch_size,
            shuffle=True
        )

    def viz_cycles(self, real_a, real_b, n=3):
        real_a = real_a[0:min(real_a.shape[0], n)]
        real_b = real_b[0:min(real_b.shape[0], n)]
        with torch.no_grad():
            fake_b = self.cyclegan.gen_b(real_a)
            fake_ba = self.cyclegan.gen_a(fake_b)

            fake_a = self.cyclegan.gen_a(real_b)
            fake_ab = self.cyclegan.gen_b(fake_a)

        a_rows = torch.cat([real_a, fake_b, fake_ba], dim=3)
        b_rows = torch.cat([real_b, fake_a, fake_ab], dim=3)
        a_img = (torch.cat(list(a_rows), dim=1) + 1.0) / 2.0
        b_img = (torch.cat(list(b_rows), dim=1) + 1.0) / 2.0
        a_img = a_img.detach().cpu().numpy()
        b_img = b_img.detach().cpu().numpy()

        return a_img, b_img

    def load_model(self, epoch, directory="checkpoint"):
        self.cyclegan.gen_a.load_state_dict(torch.load("./{}/{:05d}-gen-a".format(directory, epoch)))
        self.cyclegan.gen_b.load_state_dict(torch.load("./{}/{:05d}-gen-b".format(directory, epoch)))
        self.cyclegan.dsc_a.load_state_dict(torch.load("./{}/{:05d}-dsc-a".format(directory, epoch)))
        self.cyclegan.dsc_b.load_state_dict(torch.load("./{}/{:05d}-dsc-b".format(directory, epoch)))

    def save_model(self, epoch):
        torch.save(self.cyclegan.gen_a.state_dict(), "./checkpoint/{:05d}-gen-a".format(epoch))
        torch.save(self.cyclegan.gen_b.state_dict(), "./checkpoint/{:05d}-gen-b".format(epoch))
        torch.save(self.cyclegan.dsc_a.state_dict(), "./checkpoint/{:05d}-dsc-a".format(epoch))
        torch.save(self.cyclegan.dsc_b.state_dict(), "./checkpoint/{:05d}-dsc-b".format(epoch))

    @staticmethod
    def _get_labeled_indices(dataset, class_names):
        indices = []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] in class_names:
                indices.append(i)
        return indices

    def train(self, epochs, save_every=100, test_every=50):
        train_a = iter(self.load_a)
        train_b = iter(self.load_b)
        
        for epoch in range(epochs):
            print(f"\rEpoch {epoch}/{epochs}\t", end="")

            # Fetch data
            try:
                batch_a, _ = next(train_a)
                batch_b, _ = next(train_b)
            except StopIteration:
                train_a = iter(self.load_a)
                train_b = iter(self.load_b)
                batch_a, _ = next(train_a)
                batch_b, _ = next(train_b)

            # To GPU & scale
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)
            batch_a = batch_a * 2.0 - 1.0
            batch_b = batch_b * 2.0 - 1.0

            # Train
            self.cyclegan.epoch(batch_a, batch_b)

            # Output image
            if epoch % test_every == 0:
                a_img, b_img = self.viz_cycles(batch_a, batch_b, n=3)
                self.metrics.record_image("viz/a_cycle", a_img)
                self.metrics.record_image("viz/b_cycle", b_img)

            # Save model
            if epoch % save_every == 0 or epoch == epochs - 1:
                self.save_model(epoch)
                print("\tSaved checkpoint at", epoch, "\t", end="")

            self.metrics.flush(epoch)


if __name__ == "__main__":

    print("Running CycleGAN...")
    print("Cuda available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device: ", torch.cuda.get_device_name(0))

    t = Trainer(batch_size=1)
    t.train(1000000, save_every=1000, test_every=100)


