#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import csv
import datetime
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import CNNArchitectures


class LeafCountDataset(Dataset):
    
    def __init__(self, images_filepath, labels_filepath, mask, transform=None, grayscale=False):
        self.transform = transform
        self.grayscale = grayscale
        
        data = np.genfromtxt(labels_filepath, delimiter=',')[1:,][mask]
        self.image_ids = np.array([int(n) for n in data[:,0]])
        self.minima = data[:,1]
        self.maxima = data[:,2]
        self.averages = np.apply_along_axis(lambda x: (x[1]+x[2])/2, axis=1, arr=data)
        
        # Normalize inputs to [0, 1]
        if grayscale:
            self.images = np.array([np.array(Image.open('{}/{}.png'.format(images_filepath, n)).convert('L'))/255 for n in self.image_ids])
        else: 
            self.images = np.array([np.array(Image.open('{}/{}.png'.format(images_filepath, n)))/255 for n in self.image_ids])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if self.grayscale:
            image = self.images[index].reshape((256, 256, 1))
        else:
            image = self.images[index].reshape((256, 256, 3))
        label = self.averages[index]
        minimum = self.minima[index]
        maximum = self.maxima[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image_id, image, label, minimum, maximum


class Experiment:
    def __init__(self, net, LR, DROPOUT, BATCH_SIZE, DECAY):
        # Set hyperparameters
        self.LR = LR
        self.DROPOUT = DROPOUT
        self.BATCH_SIZE = BATCH_SIZE
        self.DECAY = DECAY
        
        # Instantiate model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.model = self.net(self.DROPOUT).to(self.device)
        self.architecture = net.__name__
        
        # Set loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=self.DECAY) 
        
        # Make training log variables
        self.training_log = []
        self.epochs_trained = 0

    def training(self, dataset_train, NUM_EPOCHS):
        # Make dataloader from dataset
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.BATCH_SIZE, shuffle=True)
        
        # Make training log variables
        training_start = datetime.datetime.now()
        loss_log = {}
        total_step = len(train_loader)
        
        for epoch in range(1, NUM_EPOCHS+1):
            loss_for_epoch = []
            loss_for_step = 0
            for i, (image_ids, images, labels, lower_bounds, upper_bounds) in enumerate(train_loader): 
                # Convert tensors to proper type depending on device type
                if torch.cuda.is_available():
                    images = images.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.FloatTensor)
                else:
                    labels = labels.type(torch.FloatTensor)

                # Pass data 
                images.to(self.device)
                labels.to(self.device)

                # Forward pass
                if torch.cuda.is_available():
                    outputs = self.model(images).type(torch.cuda.FloatTensor)
                    outputs = torch.squeeze(outputs)
                else:
                    outputs = self.model(images)

                # Calculate loss
                loss = self.criterion(outputs, labels)
                loss_for_epoch.append(loss.item())

                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_for_step += loss.item()

                # Show progress
                if (i+1) % 100 == 0:
                    loss_for_step /= 100
                    print('Epoch [{} / {}], Step [{} / {}], Average Loss: {}'.format(epoch+self.epochs_trained, NUM_EPOCHS+self.epochs_trained, i+1, total_step, loss_for_step))
                    loss_for_step = 0
                    
            loss_log[epoch] = (loss_for_epoch, datetime.datetime.now())
        
        training_end = datetime.datetime.now()
        self.training_log.append((training_start, training_end, loss_log))
        self.epochs_trained += NUM_EPOCHS
        
    def testing(self, dataset_test):
        # Make dataloader from dataset
        test_loader = DataLoader(dataset=dataset_test, batch_size=self.BATCH_SIZE, shuffle=False)
        
        # Make testing log variables
        self.predictions = []
        self.errors = []
        self.image_ids_test = []
        self.minima_test = []
        self.maxima_test = []
        
        for i, (img_ids, images, labels, lower_bounds, upper_bounds) in enumerate(test_loader):
            # Convert tensors to proper type depending on device type
            if torch.cuda.is_available():
                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)
            else:
                labels = labels.type(torch.FloatTensor)
            
            # Pass data
            images.to(self.device)

            # Forward pass and save predictions
            if torch.cuda.is_available():
                outputs = self.model(images).type(torch.cuda.FloatTensor)
                outputs = torch.squeeze(outputs)
                outputs = outputs.tolist()
                if type(outputs) == float: outputs = [outputs]
                self.predictions += [int(round(pred)) for pred in outputs] 
                self.errors += [abs(int(round(pred))-labels.tolist()[j]) for j, pred in enumerate(outputs)]
            else:
                outputs = self.model(images)
                outputs = outputs.tolist()
                self.predictions += [int(round(pred[0])) for pred in outputs] 
                self.errors += [abs(int(round(pred[0]))-labels.tolist()[j]) for j, pred in enumerate(outputs)]
            
            # Save image_ids, minima, and maxima 
            self.image_ids_test += [img_id for img_id in img_ids.tolist()]
            self.minima_test += [mini for mini in lower_bounds.tolist()]
            self.maxima_test += [maxi for maxi in upper_bounds.tolist()]
            
        # Calculate average error and accuracy
        self.average_error = np.mean(self.errors)
        self.accuracy = sum([1 for i, pred in enumerate(self.predictions) if (pred >= self.minima_test[i]) and (pred <= self.maxima_test[i])])/len(self.predictions)

            
    def save(self, SAVE_DIR=os.path.join(os.path.dirname(__file__), '../../models')):

        # Save predictions 
        with open(SAVE_DIR+'/'+'predictions_{}_LR={}_DROPOUT={}_BATCH_SIZE={}_DECAY={}.csv'.format(self.architecture, self.LR, self.DROPOUT, self.BATCH_SIZE, self.DECAY), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'prediction', 'actual range', 'error'])
            for i, image_id in enumerate(self.image_ids_test):
                writer.writerow([str(self.image_ids_test[i]), str(self.predictions[i]), '{} , {}'.format(int(self.minima_test[i]), int(self.maxima_test[i])), str(self.errors[i])])

        # Save experiment information
        with open(SAVE_DIR+'/'+'experiment_{}_LR={}_DROPOUT={}_BATCH_SIZE={}_DECAY={}.txt'.format(self.architecture, self.LR, self.DROPOUT, self.BATCH_SIZE, self.DECAY), 'w') as f:
            f.write('epochs trained: {} \n \n'.format(self.epochs_trained))
            f.write('average error: {} \n \n'.format(self.average_error))
            f.write('accuracy: {} \n \n'.format(self.accuracy))
            f.write('training log: {} \n \n'.format(self.training_log))
        
        # Save model
        torch.save(self.model.state_dict(), SAVE_DIR+'/'+'model_{}_LR={}_DROPOUT={}_BATCH_SIZE={}_DECAY={}.pt'.format(self.architecture, self.LR, self.DROPOUT, self.BATCH_SIZE, self.DECAY))


def get_datasets(images_path, labels_path, split_ratio=0.85, grayscale=False):
    # Define transforms 
    transformations = transforms.Compose([transforms.ToTensor()])

    # Define mask for train/test split
    num_plants = sum([1 for f in os.listdir(images_path) if '.png' in f])
    mask = np.random.rand(num_plants) <= split_ratio

    # Make grayscale datasets
    dataset_train = LeafCountDataset(images_path, labels_path, mask, transformations, grayscale=grayscale)
    dataset_test = LeafCountDataset(images_path, labels_path, np.logical_not(mask), transformations, grayscale=grayscale)

    return (dataset_train, dataset_test)


def find_best_hyperparameters(dataset_train, dataset_test, architectures, learning_rates, dropouts, batch_sizes, decays, num_epochs=5):
    best_architecture = None
    best_lr = None
    best_dropout = None
    best_batch_size = None
    best_decay = None
    best_accuracy = -float('inf')

    for architecture in architectures:
        for lr in learning_rates:
            for dropout in dropouts:
                for batch_size in batch_sizes:
                    for decay in decays:
                        print('Trying: architecture={}, learning_rate={}, dropout={}, batch_size={}, decay={}'.format(architecture,lr,dropout,batch_size,decay))
                        exp = Experiment(architecture, LR=lr, DROPOUT=dropout, BATCH_SIZE=batch_size, DECAY=decay)
                        exp.training(dataset_train, num_epochs)
                        exp.testing(dataset_test)
                        print('Accuracy: {}'.format(round(exp.accuracy, 3)))
                        if exp.accuracy > best_accuracy:
                            best_accuracy = exp.accuracy
                            best_architecture = architecture
                            best_lr = lr
                            best_batch_size = batch_size
                            best_dropout = dropout
                            best_decay = decay

    'Best hyperparameters: architecture={}, learning_rate={}, dropout={}, batch_size={}, decay={}'.format(best_architecture,best_lr,best_dropout,best_batch_size,best_decay)

    return (best_architecture, best_lr, best_dropout, best_batch_size, best_decay)


def main():

    # File paths
    DIR = os.path.dirname(__file__)
    IMAGES_PATH = os.path.join(DIR, '../../data/processed/images')
    LABELS_PATH = os.path.join(DIR, '../../data/processed/labels/labels.csv')

    # Hyperparameters
    ARCHITECTURES_GRAY = [CNNArchitectures.Conv4Net_1Channel_Narrow, CNNArchitectures.Conv4Net_1Channel_Wide, 
                          CNNArchitectures.Conv5Net_1Channel_Narrow, CNNArchitectures.Conv5Net_1Channel_Wide]
    ARCHITECTURES_COLOR = [CNNArchitectures.Conv6Net_3Channel_Narrow, CNNArchitectures.Conv6Net_3Channel_Wide,
                           CNNArchitectures.Conv7Net_3Channel_Narrow, CNNArchitectures.Conv7Net_3Channel_Wide]
    NUM_EPOCHS = 30
    LEARNING_RATES = [1e-5, 1e-4, 5e-5, 5e-4]
    DROPOUTS = [0.1, 0.3, 0.5]
    BATCH_SIZES = [1, 4, 16]
    DECAYS = [0, 1e-4, 1e-2]

    # Make datasets
    dataset_train_gray, dataset_test_gray = get_datasets(IMAGES_PATH, LABELS_PATH, grayscale=True)
    dataset_train_color, dataset_test_color = get_datasets(IMAGES_PATH, LABELS_PATH, grayscale=False)

    # Determine best hyperparameters 
    architecture_gray, lr_gray, dropout_gray, batch_size_gray, decay_gray = find_best_hyperparameters(dataset_train_gray, dataset_test_gray, ARCHITECTURES_GRAY, LEARNING_RATES, DROPOUTS, BATCH_SIZES, DECAYS)
    architecture_color, lr_color, dropout_color, batch_size_color, decay_color = find_best_hyperparameters(dataset_train_color, dataset_test_color, ARCHITECTURES_COLOR, LEARNING_RATES, DROPOUTS, BATCH_SIZES, DECAYS)

    # Train and save gray model
    exp_gray = Experiment(architecture_gray, LR=lr_gray, DROPOUT=dropout_gray, BATCH_SIZE=batch_size_gray, DECAY=decay_gray)
    exp_gray.training(dataset_train_gray, NUM_EPOCHS=NUM_EPOCHS)
    exp_gray.testing(dataset_test_gray)
    exp_gray.save()

    # Train and save color model
    exp_color = Experiment(architecture_color, LR=lr_color, DROPOUT=dropout_color, BATCH_SIZE=batch_size_color, DECAY=decay_color)
    exp_color.training(dataset_train_color, NUM_EPOCHS=NUM_EPOCHS)
    exp.testing(dataset_test_color)
    exp_color.save_()


if __name__ == '__main__':
    main()



