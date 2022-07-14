""" Training, validation and predictions for classification and segmentation tasks. """

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import numpy as np
import math
import torchmetrics

class Train:
    """ Train, validation and testing model for classification or
    segmenatation task. Plot losses and metrics results.
    """
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
        # save values of average loss for every epoch 
        self.avg_loss_train = []
        self.avg_loss_valid = []
        self.accuracy_valid = []
        self.iou_valid = []
    
    def get_train_avg_loss(self) -> list:
        """ Get saved average train loss for every epoch. """
        return self.avg_loss_train
    
    def get_valid_avg_loss(self) -> list:
        """ Get saved average validation loss for every epoch. """
        return self.avg_loss_valid
    
    def get_valid_accuracy(self) -> list:
        """ Get saved average validation accuracy for every epoch. """
        return self.accuracy_valid
    
    def get_valid_iou(self) -> np.array:
        """ Get saved average validation IoU score for every epoch. """
        return self.iou_valid
    
    def training_one_epoch(self):
        train_losses = []
        # Training
        self.model.train(True)
        running_loss = 0.

        for i, data in enumerate(self.train_loader):
            
            inps, labels = data
            inps = inps.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(inps)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / (i + 1)
        self.avg_loss_train.append(avg_loss)
        
        return avg_loss
    
    def val_segmentation_one_epoch(self):
        running_loss = 0.
        iou_score = np.empty(len(self.val_loader))

        jaccard = torchmetrics.JaccardIndex(num_classes=151).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inps, labels_cpu = data

                inps = inps.to(self.device)
                masks = labels_cpu.to(self.device)

                outputs = self.model(inps)
                predicted = torch.argmax(outputs, 1)
                
                loss = self.loss_function(outputs, masks)
                running_loss += loss.item()
                
                # IoU score
                val_iou = jaccard(predicted, masks).cpu().numpy()
                iou_score[i] = val_iou  
                    
        avg_loss = running_loss / (i + 1)
        self.avg_loss_valid.append(avg_loss)
        self.iou_valid.append(np.mean(iou_score))
        
        return avg_loss
    
    @torch.no_grad()
    def val_classification_one_epoch(self, action: str='validation'):
        """
        Parameters
        ----------
        action:  str
            Validation or testing
        """
        loader = self.val_loader
        if action == 'testing':
            loader = self.test_loader
            
        running_loss = 0.
        self.model.eval()
        total = 0.
        correct = 0.
        running_vloss = 0.

        for i, data in enumerate(loader):
            inps, labels_cpu = data

            inps = inps.to(self.device)
            labels = labels_cpu.to(self.device)
                
            outputs = self.model(inps)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
            if action == 'validation':
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()
        
        # calculate f1 score for one epoch  
        f1score = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
        # calculating an accuracy for the all batches (one epoch)
        temp_accuracy = 100 * correct / total
        
        if action == 'validation':       
            # calculate average loss per epoch
            avg_loss = running_loss / (i + 1)
            self.avg_loss_valid.append(avg_loss)
            self.accuracy_valid.append(temp_accuracy)
            
            return temp_accuracy, avg_loss
    
        return temp_accuracy, f1score
        
    def train_procedure(self, task='classification'):
        """
        Parameters
        ----------
        task: str
            Classification or segmentation
            metrics: accuracy and f1 score for classification.
                     IoU score for segmentation.
        """
        if task == 'classification':
            
            for epoch in range(self.num_epochs):
                print('Epoch number is', epoch + 1)
                # TRAINING
                avg_loss = self.training_one_epoch()
            
                # VALIDATION
                temp_accuracy, avg_vloss = self.val_classification_one_epoch(action='validation')
            # TESTING
            temp_accuracy_test, f1_score_test = self.val_classification_one_epoch(action='testing')

            return round(temp_accuracy_test, 3), round(f1_score_test, 3)
            
        elif task == 'segmentation':
            for epoch in range(self.num_epochs):
                print('Epoch number is', epoch + 1)
                # TRAINING
                avg_loss = self.training_one_epoch()
                # VALIDATION
                avg_vloss = self.val_segmentation_one_epoch()
    
    def predict_segmentation(self, idx, fig_widtht=10, fig_height=10):
        """ Predict image mask for semantic segmentation task.
        
        Parameters
        ----------
        idx: int
            plt.show image, its true mask and predicted mask by idx
        """
        self.model.eval()   

        inputs, labels = next(iter(self.test_loader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(inputs)
        
        predicted = torch.argmax(outputs, axis=1)
        target_masks = labels.cpu().numpy()
        predicted = predicted.cpu().detach().numpy()
        
        for i in range(idx, idx + 3):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_widtht, fig_height))

            ax1.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
            ax1.set_title('source image')

            ax2.imshow(target_masks[i])
            ax2.set_title('ground truth')

            ax3.imshow(predicted[i])
            ax3.set_title('predicted mask')
            plt.show()
            
    
    def plot_losses(self, train_losses: list, valid_losses: list,
                    fig_size_width: int, fig_size_height: int):
        
        plt.figure(figsize=(fig_size_width, fig_size_height))
        plt.plot(train_losses, label='Train')
        plt.plot(valid_losses, label='Validation')
        plt.legend()
        plt.title('Train and validation losses')
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    def plot_metric(self, validation_accuracy: list,
                      fig_size_width: int, fig_size_height: int, metric_name='Accuracy'):
        
        plt.figure(figsize=(fig_size_width, fig_size_height))
        plt.plot(validation_accuracy, label='Validation')
        plt.legend()
        plt.title(f'Validation {metric_name.lower()}')
        plt.xlabel('Epoch number')
        plt.ylabel(f'{metric_name}')
        plt.grid(True)
        plt.show()
    
    def show_sample(self, img, target):
        plt.imshow(img.permute(1, 2, 0))
        print('Labels:', target)
        
    def predict_image(self, img):

        xb = img.unsqueeze(0)
        xb = xb.to(self.device)
        yb = self.model(xb)
        _, preds  = torch.max(yb, dim=1)

        self.show_sample(img, target=self.test_dataset.classes[preds[0].item()])
        
    def plot_confusion_matrix(self, fig_size_width=10, fig_size_height=8):
        predictions = np.array([0])
        labels_true = np.array([0])

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.test_loader):
                imgs = imgs.to(self.device)
                labels = labels.numpy().astype(int)
                predictions_batch = self.model(imgs).argmax(axis=1).detach().cpu().numpy().astype(int)

                predictions = np.concatenate((predictions, predictions_batch), axis=None)
                labels_true = np.concatenate((labels_true, labels), axis=None)
        
        predictions, labels_true = predictions[1:], labels_true[1:]
        
        plt.figure(figsize=(fig_size_width, fig_size_height))
        cnf_matrix = confusion_matrix(labels_true, predictions)
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix,
                                                display_labels=self.test_dataset.classes)
        display_matrix.plot()
        plt.xticks(rotation=90)
        plt.show()

        

