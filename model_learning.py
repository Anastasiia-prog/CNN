# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

class Training_valid_testing_model:
    
    def __init__(self, model, loss_function, optimizer, num_epochs,
                 train_loader, val_loader, test_loader, test_dataset, device):
        
        self.model = model
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.device = device
    
    def calculate_temp_accuracy(self, total, correct):
        '''
        Calculating an accuracy for the all batches (one epoch)
        '''
        return 100 * correct / total
    
    def calculate_avg_loss(self, running_loss, i):
        '''
        Calculating an average loss for the all batches (one epoch)
        '''
        return running_loss / (i + 1)
    
    def training_one_epoch(self):
        train_losses = []
        # Training
        self.model.train(True)
        running_loss = 0.

        for i, data in tqdm(enumerate(self.train_loader)):
            inps, labels = data

            inps = inps.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(inps)
            loss = self.loss_function(output, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        avg_loss = self.calculate_avg_loss(running_loss, i)
        
        return running_loss, avg_loss
            
    def validation_one_epoch(self):
        running_vloss = 0.
        self.model.train(False)
        with torch.no_grad():
            total = 0.
            correct = 0.
            running_vloss = 0.

            for i, vdata in tqdm(enumerate(self.val_loader)):
                vinps, vlabels = vdata

                vinps = vinps.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self.model(vinps)

                _, predicted = torch.max(voutputs.data, 1)
            
                total += vlabels.size(0)
                correct += (predicted == vlabels).sum().item()

                vloss = self.loss_function(voutputs, vlabels)
                running_vloss += vloss.item()
                
        # calculate average loss per epoch
        avg_vloss = self.calculate_avg_loss(running_vloss, i)
        # calculate accuracy for one epoch
        temp_accuracy = self.calculate_temp_accuracy(total, correct)
    
        return running_vloss, avg_vloss, temp_accuracy
    
    def testing(self):
        # Testing
        self.model.train(False)
        num_correct = 0.
        num_samples = 0.

        for batch_idx, (data, labels) in enumerate(self.test_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            scores = self.model(data)
    
            _, predicted = torch.max(scores.data, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()

            f1_score = torch.div(num_correct, len(labels))
            test_accuracy = num_correct / num_samples * 100
            
        return test_accuracy
    
    def plot_losses(self, train_losses, valid_losses):
        plt.plot(train_losses, label='Train')
        plt.plot(valid_losses, label='Validation')
        plt.legend()
        plt.title('Графики лоссов на train и validation')
        plt.xlabel('Номер эпохи')
        plt.ylabel('Значение лосса')
        plt.grid(True);
    
    def plot_accuracy(self, validation_accuracy):
        plt.plot(validation_accuracy, label='Validation')
        plt.legend()
        plt.title('Validation accuracy')
        plt.xlabel('Номер эпохи')
        plt.ylabel('Значение accuracy')
        plt.grid(True);
        
    def learning_run(self):
        train_losses = []
        valid_losses = []
        validation_accuracy = []

        for epoch in range(self.num_epochs):
            print('Epoch number is', epoch + 1)
            # TRAINING
            running_loss, avg_loss = self.training_one_epoch()

            # save values of average loss for every epoch 
            train_losses.append(avg_loss)

            # VALIDATION
            running_vloss, avg_vloss, temp_accuracy = self.validation_one_epoch()

            # save values of average loss for every epoch 
            valid_losses.append(avg_vloss)
            # save accuracy for every epoch
            validation_accuracy.append(temp_accuracy)

            print(f'CrossEntropyLoss train {avg_loss}')
            print(f'CrossEntropyLoss validation {avg_vloss}')
            print(f'Validation accuracy: {temp_accuracy}')
            
        self.plot_losses(train_losses, valid_losses)
        plt.show()
        self.plot_accuracy(validation_accuracy)
        plt.show()
    
        
    def show_sample(self, img, target):
        plt.imshow(img.permute(1, 2, 0))
        print('Labels:', target)
        
    def predict_image(self, img):

        xb = img.unsqueeze(0)
        xb = xb.to(self.device)
        yb = self.model(xb)
        _, preds  = torch.max(yb, dim=1)
        
        print('Labels:', target)
        show_sample(img, self.test_dataset.classes[preds[0].item()])
        
    def testing_results(self):
        test_accuracy = self.testing()

