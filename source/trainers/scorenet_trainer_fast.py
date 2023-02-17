from source.utils.utils import correct_predictions, rand_bbox, scoremix_bbox
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from source.utils.utils import accuracy
from einops import rearrange
import pandas as pd
import numpy as np
import torch
import wandb
import os


class ScoreNetTrainerFast:
    def __init__(self, model, classifier, train_loader, valid_loader, test_loader, savepath, args, logger,
                 loadpath=None, display_rate=10, decay_noise=True, fp16_scaler=None, mix_prob=0.5,
                 test_freq=10, downscaling_ratio=8, augmentation='scoremix', load_model_only=True):
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.model = model.to(self.device)
        self.classifier = classifier.to(self.device)
        self.arch = args.arch

        # Data generators
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Optimization parameters
        self.criterion = CrossEntropyLoss()
        self.criterion_no_reduce = CrossEntropyLoss(reduce=False)

        # Misc.
        self.logger = logger
        self.mix_prob = mix_prob
        self.test_freq = test_freq
        self.fp16_scaler = fp16_scaler
        self.decay_noise = decay_noise
        self.display_rate = display_rate
        self.augmentation = augmentation
        self.downscaling_ratio = downscaling_ratio

        # Loss function and stored losses
        self.best_valid_f1_score = 0.
        self.losses = {'train': [], 'valid': []}
        self.batch_losses = {'train': [], 'valid': []}
        self.accuracies = {'train': [], 'valid': []}
        self.f1_scores = {'weighted': [], 'class': []}
        self.sigma_ratios = []

        # Path to save to the class
        self.savepath = savepath
        self.loadpath = loadpath

        # Load saved states
        if loadpath is not None and os.path.exists(loadpath):
            self.load(load_model_only)
        self.current_epoch = len(self.losses['train'])

        # set the optimizer
        self.optimizer = torch.optim.SGD(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            # filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr * args.batch_size / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=1e-6)

        # Display the number of trainable parameters
        capacity = sum([p.numel() for param_group in self.optimizer.param_groups for p in param_group['params']])
        print('Total number of trainable parameters: {}'.format(capacity))

    def save(self, is_last_epoch=False):
        """
        Saves the required attributes to resume training later.
        :param is_last_epoch: Boolean flag indicating if the training is at its last epoch.
        :return: None.
        """
        if is_last_epoch:
            savepath = '.'.join(self.savepath.split('.')[:-1]) + '_final.pth'
        else:
            savepath = '.'.join(self.savepath.split('.')[:-1]) + '_{}.pth'.format(self.current_epoch)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'losses_batch': self.batch_losses,
            'accuracies': self.accuracies,
            'f1_scores': self.f1_scores,
            'best_valid_f1_score': self.best_valid_f1_score,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'sigma': self.model.perturbed_topk.sigma if self.arch == 'scorenet' else None
        }
        if self.fp16_scaler is not None:
            save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()
        torch.save(save_dict, savepath)
        self.logger.debug('Saved weights!')

    def load(self, load_model_only):
        """
        Loads the model, optimizer, losses, and queue.
        :return: None.
        """
        checkpoint = torch.load(self.loadpath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('Pre-trained model loaded')
        if not load_model_only:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.losses = checkpoint['losses']
            self.batch_losses = checkpoint['losses_batch']
            self.accuracies = checkpoint['accuracies']
            self.f1_scores = checkpoint['f1_scores']
            self.best_valid_f1_score = checkpoint['best_valid_f1_score']
            self.model.perturbed_topk.sigma = checkpoint['sigma']
            if self.fp16_scaler is not None:
                self.fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

    @staticmethod
    def reset_gradients(model):
        """
        Resets the gradient without reading it.
        :return: None
        """
        for p in model.parameters():
            p.grad = None

    @staticmethod
    def change_grad_status(requires_grad, model):
        """
        Set the requires flag status of the provided model.
        :param requires_grad: flag indicating if the model needs to track gradients.
        :param model: model for which the status requires a change.
        :return: None.
        """
        # Iterate over the model's parameters
        for p in model.parameters():
            p.requires_grad = requires_grad

    def evaluate(self):
        """
        Evaluates the performance of the model on the validation set.
        :param: last epoch of the session.
        :return: None.
        """
        # Set both models on evaluation mode
        self.model.eval()
        self.classifier.eval()

        # Iterate over the batches in the validation set
        batch_losses, outputs, targets = [], [], []
        for batch in self.valid_loader:
            # Move data to device
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            # Store the targets
            targets.append(labels.cpu())

            # Get patch level predictions
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                output = self.model(images, True)
                output = self.classifier(output)

            # Compute the loss and the accuracy
            loss = self.criterion(output, labels).item()
            batch_losses.append(loss)
            self.batch_losses['valid'].append(loss)

            # Store the predictions
            outputs.append(output.cpu())

        # Compute the F1-score
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        current_accuracy, = accuracy(outputs, targets, topk=(1,))
        outputs = torch.argmax(outputs, dim=-1)
        self.f1_scores['weighted'].append(f1_score(targets, outputs, average='weighted'))
        self.f1_scores['class'].append(f1_score(targets, outputs, average=None))

        # Store the results
        self.losses['valid'].append(np.mean(batch_losses))
        self.accuracies['valid'].append(current_accuracy)

    def cutmix_data(self, images, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        # generate mixed sample
        batch_size = images.shape[0]
        index = torch.randperm(batch_size)
        index = index.to(self.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        return images, y_a, y_b, lam

    def mixup_data(self, images, y, alpha=1.0):
        """
        Returns mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = images.shape[0]
        index = torch.randperm(batch_size)
        index = index.to(self.device)

        mixed_images = lam * images + (1. - lam) * images[index, :]
        y_a, y_b = y, y[index]
        return mixed_images, y_a, y_b, lam

    def score_mix_data(self, images, thumbnails, y, alpha=1.0):
        """
        Returns mixed inputs, pairs of targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = len(thumbnails)
        index = torch.randperm(batch_size)
        index = index.to(self.device)

        # Get the saliency maps
        with torch.no_grad():
            saliency_maps = self.model.get_saliency_maps(thumbnails)

        # Get the positive masks
        positive_indices, negative_indices, lam = scoremix_bbox(lam, saliency_maps, index)

        # Patchify the images
        small_patch_size = self.model.small_patch_size
        large_patch_size = self.model.large_patch_size

        # Get the source and target thumbnails
        thumbnails_target, thumbnails_source = thumbnails.clone(), thumbnails
        thumbnails_target = rearrange(thumbnails_target, ' b c (n h) (m w) -> b c n m h w', h=small_patch_size,
                                      w=small_patch_size)
        thumbnails_source = rearrange(thumbnails_source, 'b c (n h) (m w) -> b c n m h w', h=small_patch_size,
                                      w=small_patch_size)

        # Get the source and target images
        images_target, images_source = images.clone(), images
        images_target = rearrange(images_target, 'b c (n h) (m w) -> b c n m h w', h=large_patch_size,
                                  w=large_patch_size)
        images_source = rearrange(images_source, 'b c (n h) (m w) -> b c n m h w', h=large_patch_size,
                                  w=large_patch_size)

        # Paste the patch
        thumbnails_target[negative_indices] = thumbnails_source[positive_indices]
        images_target[negative_indices] = images_source[positive_indices]

        # Move image and thumbnail back to original shape
        thumbnails_target = rearrange(thumbnails_target, 'b c n m h w -> b c (n h) (m w)')
        images_target = rearrange(images_target, 'b c n m h w -> b c (n h) (m w)')

        # Return the new signal and image pairs
        y_a, y_b = y, y[index]
        return images_target, thumbnails_target, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        if isinstance(lam, list):
            n = len(lam)
            lam_ = torch.tensor(lam).to(pred.device).to(pred.dtype)
            loss_a = self.criterion_no_reduce(pred, y_a)
            loss_b = self.criterion_no_reduce(pred, y_b)
            loss_a, loss_b = torch.einsum('b, b ->', loss_a, lam_), torch.einsum('b, b ->', loss_b, 1. - lam_)
            return (loss_a + loss_b) / n
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_step(self, batch):
        """
        Executes a single step of training of the PAWS framework on the provided batch.
        :param batch: single batch of data.
        :return: losses and accuracy.
        """
        # Move data to device
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        # Apply the regularization method
        r = np.random.rand(1)
        if r <= self.mix_prob:
            r = np.random.rand(1)
            if r <= 0.5:
                images, labels_a, labels_b, lam = self.mixup_data(images, labels)
            else:
                images, labels_a, labels_b, lam = self.cutmix_data(images, labels)
        else:
            labels_a = labels_b = labels
            lam = 0.5

        # Get patch level predictions
        with torch.cuda.amp.autocast(self.fp16_scaler is not None):
            outputs = self.model(images, False)
            outputs = self.classifier(outputs)

        _, predicted = torch.max(outputs.data, 1)

        # Compute the loss as a convex combination
        # loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
        loss = self.criterion(outputs, labels)

        # Compute the # correct predictions as a convex combination
        if isinstance(lam, list):
            lam_ = torch.tensor(lam).to(labels.device).to(float)
            n_corrects_a = predicted.eq(labels_a).to(float)
            n_corrects_b = predicted.eq(labels_b).to(float)
            n_corrects_a = torch.einsum('b, b ->', n_corrects_a, lam_)
            n_corrects_b = torch.einsum('b, b ->', n_corrects_b, 1 - lam_)
            n_corrects = (n_corrects_a + n_corrects_b).cpu()
        else:
            n_corrects = (lam * predicted.eq(labels_a.data).cpu().sum().float() +
                          (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())

        # Get the number of processed samples
        n_samples = outputs.shape[0]

        # Back-propagate the loss
        if self.fp16_scaler is None:
            self.optimizer.zero_grad()
            loss.backward()

            # Update the model's weights
            self.optimizer.step()
        else:
            self.fp16_scaler.scale(loss).backward()

            # Update the model's weights
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        return loss.item(), n_samples, n_corrects

    def train(self, epochs):
        """
        Trains the models for a specified number of epochs.
        :param epochs: number of iterations over the training dataset to perform.
        :return: None.
        """
        # Start the wandb logger
        wandb.watch(self.model, self.criterion, log='all', log_freq=1)

        # Display start status
        self.logger.debug('About to train for {} epochs.'.format(epochs))

        # Infer the end epoch of the current session
        end_epoch = self.current_epoch + epochs - 1
        for epoch in range(epochs):
            # Set the models in train mode
            self.model.train()
            self.classifier.train()

            # Initialize metrics, etc.
            batch_losses, batch_ratios, batch_accuracies = [], [], []
            total_samples, total_corrects = 0, 0
            for j, batch in enumerate(self.train_loader):
                # Execute a training step
                loss, n_samples, n_corrects = self.train_step(batch)

                # Compute the accuracy
                total_samples += n_samples
                total_corrects += n_corrects
                batch_accuracies.append(n_corrects.item() / n_samples)

                # Store the batch losses
                batch_losses.append(loss)
                try:
                    batch_ratios.append(self.model.ratio.item())
                except AttributeError:
                    pass
                self.batch_losses['train'].append(loss)

                # Display status
                if j % self.display_rate == 0:
                    n_batches = len(self.train_loader.dataset) // self.train_loader.batch_size
                    message = 'epoch: {}/{}, batch {}/{}: \n' \
                              '\t loss: {:.2e}, acc: {:.2e}, sigma: {:.2e}, ratio: {:.2e}, lr: {:.2e}' \
                        .format(self.current_epoch, end_epoch, j, n_batches,
                                np.mean(batch_losses[-self.display_rate:]),
                                np.mean(batch_accuracies),
                                self.model.perturbed_topk.sigma if self.arch == 'scorenet' else 0.,
                                self.model.ratio if self.arch == 'scorenet' else 0.,
                                self.optimizer.param_groups[0]["lr"])
                    self.logger.debug(message)

            # Update the scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Store the epoch loss
            self.losses['train'].append(np.mean(batch_losses))
            self.sigma_ratios.append(np.mean(batch_ratios))

            # Compute the current epoch's accuracy
            self.accuracies['train'].append(total_corrects / total_samples)

            # Evaluate the model
            with torch.no_grad():
                self.evaluate()

            # Evaluate on the test set for the ablations
            if epoch % self.test_freq == 0 and epoch != 0:
                with torch.no_grad():
                    self.test()
                    self.save()

            # Terminate epoch
            self.on_epoch_end(end_epoch)

        # Save the final state
        self.save(True)

    def on_epoch_end(self, end_epoch):
        """
        Display the current epoch status.
        :param end_epoch: last epoch od the session.
        :return: None.
        """
        # Display epoch end message
        message = 'average metrics at epoch {}/{}: \n' \
                  '\t train: loss: {:.2e}, accuracy: {:.2e} \n' \
                  '\t valid: loss: {:.2e}, accuracy: {:.2e}, f1-score: {:.2e} \n' \
            .format(self.current_epoch, end_epoch, self.losses['train'][-1], self.accuracies['train'][-1],
                    self.losses['valid'][-1], self.accuracies['valid'][-1],
                    self.f1_scores['weighted'][-1])

        self.logger.debug(message)
        wandb_dict = {
            'Epoch': self.current_epoch,
            'Training loss': self.losses['train'][-1],
            'Validation loss': self.losses['valid'][-1],
            'Validation f1 score': self.f1_scores['weighted'][-1],
            'Training accuracy': self.accuracies['train'][-1],
            'Validation accuracy': self.accuracies['valid'][-1],
            'Sigma (noise level)': self.model.perturbed_topk.sigma if self.arch == 'scorenet' else 0.,
            'Patch ratio': self.sigma_ratios[-1]
        }

        # Add the class F1-scores
        for k, v in zip(self.test_loader.dataset.class_dict.keys(), self.f1_scores['class'][-1]):
            wandb_dict[k] = v
        wandb.log(wandb_dict)

        # Update the noise value
        if self.decay_noise and not self.model.perturbed_topk.noise_is_zero:
            self.model.perturbed_topk.update_sigma()

        # Update the epoch counter
        self.current_epoch += 1

    @staticmethod
    def make_csv_submission(predictions, cases, csv_savepath):
        """
        Save the predictions in .csv format.
        :param predictions: predictions stored in a tensor.
        :param cases: indices of the TRoIs.
        :param csv_savepath: location where to save the .csv file.
        :return: None.
        """
        data = {'case': cases.numpy(), 'class': predictions.numpy()}
        data_df = pd.DataFrame(data)
        data_df.to_csv(csv_savepath, index=False)

    def test(self):
        """
        Evaluates the performance of the model on the validation set.
        :param: result_savepath: location where to save model's predictions.
        :return: None.
        """
        # Set both models on evaluation mode
        self.model.eval()
        self.classifier.eval()

        # Display start message
        self.logger.debug('Starting the evaluation on the test set.')

        # Iterate over the batches in the validation set
        total_corrects, total_samples, predictions, targets = 0, 0, [], []
        for batch in self.test_loader:
            # Move data to device
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            # Store the targets
            targets.append(labels.cpu())

            # Get the predictions
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                outputs = self.model(images, True)
                outputs = self.classifier(outputs)

            # Store the predictions
            predictions.append(torch.argmax(outputs, dim=-1).cpu())

            # Compute the accuracy
            total_corrects += correct_predictions(outputs, labels)
            total_samples += outputs.shape[0]

        # Compute the F1-scores
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        weighted_test_f1_score = f1_score(targets, predictions, average='weighted')
        class_test_f1_score = f1_score(targets, predictions, average=None)

        # Display the F1-score
        message = 'test: f1-score: {:.2e}'.format(weighted_test_f1_score)
        self.logger.debug(message)

        # Add the class F1-scores
        for k, v in zip(self.test_loader.dataset.class_dict.keys(), class_test_f1_score):
            message = 'test: f1-score for class {}: {:.2e}'.format(k, v)
            self.logger.debug(message)

        wandb_dict = {
            'Epoch': self.current_epoch,
            'test f1-score': weighted_test_f1_score
        }

        # Add the class F1-scores
        for k, v in zip(self.test_loader.dataset.class_dict.keys(), class_test_f1_score):
            key = 'test_' + k
            wandb_dict[key] = v
        wandb.log(wandb_dict)

        # Store the targets and predictions to disk
        # if result_savepath.endswith('.csv'):
        #     self.make_csv_submission(predictions, targets, result_savepath)
        # else:
        #     result_dict = {'predictions': predictions, 'targets': targets}
        #     with open(result_savepath, 'wb') as f:
        #         pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
