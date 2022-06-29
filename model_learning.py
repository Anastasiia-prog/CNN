import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import math

class Train:
    
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
        
    
    def get_train_avg_loss(self) -> list:
        # return saved average train loss
        # for every epoch
        return self.avg_loss_train
    
    def get_valid_avg_loss(self) -> list:
        # return saved average validation 
        # loss for every epoch
        return self.avg_loss_valid
    
    def get_valid_accuracy(self) -> list:
        # return saved average validation
        # accuracy for every epoch
        return self.accuracy_valid
    
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

            output = self.model(inps)
            loss = self.loss_function(output, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / (i + 1)
        self.avg_loss_train.append(avg_loss)
        
        return avg_loss
    
    def val_segmentation_one_epoch(self):
        
        running_loss = 0.
        self.model.train(False)
        with torch.no_grad():
            total = 0.
            correct = 0.
            running_vloss = 0.

            for i, data in enumerate(self.val_loader):
                inps, labels_cpu = data

                inps = inps.to(self.device)
                labels = labels_cpu.to(self.device)
                
                outputs = self.model(inps)
                
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()
                
        avg_loss = running_loss / (i + 1)
        self.avg_loss_valid.append(avg_loss)
        
        return avg_loss
    
    
    def val_classification_one_epoch(self, action: str='validation'):
        '''
        param action: validation or testing
        '''
        loader = self.val_loader
        if action == 'testing':
            loader = self.test_loader
            
        running_loss = 0.
        self.model.train(False)
        with torch.no_grad():
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
    
    def plot_accuracy(self, validation_accuracy: list,
                      fig_size_width: int, fig_size_height: int):
        
        plt.figure(figsize=(fig_size_width, fig_size_height))
        plt.plot(validation_accuracy, label='Validation')
        plt.legend()
        plt.title('Validation accuracy')
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
        
    def train_procedure(self, task='classification'):
        '''
        param: task: if you solve semantic segmentation task, you don't need to
        calculate accuracy and f1 score in train_procedure (task=segmentation)
        '''
        
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
           

    def reverse_transform(self, inp):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp
    
    def predict_segmentation(self, idx):
        '''
        Predict image mask for semantic segmentation task.
        param idx: plt.show image, its true mask and predicted mask by idx
        '''

        self.model.eval()   # Set model to evaluate mode

        inputs, labels = next(iter(self.test_loader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        pred = self.model(inputs)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()

        # Change channel-order and make 3 channels for matplot
        input_images_rgb = [self.reverse_transform(x) for x in inputs.cpu()]

        target_masks_rgb = labels.cpu().numpy()

        plt.imshow(input_images_rgb[idx])
        plt.title('source image')
        plt.show()
        plt.imshow(target_masks_rgb[idx])
        plt.title('ground truth')
        plt.show()
        plt.imshow(np.squeeze(pred[0][idx]))
        plt.title('predicted mask')
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

        

