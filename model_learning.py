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
        # save values of average loss for every epoch 
        self.avg_loss_train = []
        self.avg_loss_valid = []
        self.accuracy_valid = []

    
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
    
    def validation_testing_one_epoch(self, action: str='validation'):
        '''
        param action: validation or testing
        '''
        
        running_loss = 0.
        self.model.train(False)
        with torch.no_grad():
            total = 0.
            correct = 0.
            running_vloss = 0.

            for i, vdata in enumerate(self.val_loader):
                inps, labels = vdata

                inps = inps.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inps)

                _, predicted = torch.max(outputs.data, 1)
            
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if action == 'validation':
                    loss = self.loss_function(outputs, labels)
                    running_loss += loss.item()
            
        # calculate f1 score for one epoch  
        f1_score = torch.div(correct, len(labels))   
        # calculating an accuracy for the all batches (one epoch)
        temp_accuracy = 100 * correct / total
        
        if action == 'validation':       
            # calculate average loss per epoch
            avg_loss = running_loss / (i + 1)
            self.avg_loss_valid.append(avg_loss)
            self.accuracy_valid.append(temp_accuracy)
            
            return temp_accuracy, avg_loss
    
        return temp_accuracy, f1_score
    
    def plot_losses(self, train_losses: list, valid_losses: list,
                    fig_size_width: int, fig_size_height: int):
        
        plt.figure(figsize=(fig_size_width, fig_size_height))
        plt.plot(train_losses, label='Train')
        plt.plot(valid_losses, label='Validation')
        plt.legend()
        plt.title('Train and validation cross-entropy losses')
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
        
    def launch_epochs_calculations(self):

        for epoch in tqdm(range(self.num_epochs)):
            print('Epoch number is', epoch + 1)
            # TRAINING
            avg_loss = self.training_one_epoch()

            # VALIDATION
            temp_accuracy, avg_vloss = self.validation_testing_one_epoch(action='validation')

            print(f'CrossEntropyLoss train {avg_loss}')
            print(f'CrossEntropyLoss validation {avg_vloss}')
            print(f'Validation accuracy: {temp_accuracy}')
        
        # TESTING
        temp_accuracy_test, f1_score_test = self.validation_testing_one_epoch(action='testing')
        print('Test accuracy:', temp_accuracy_test)
        print('Test F1-score:', f1_score_test)
    
    def show_sample(self, img, target):
        plt.imshow(img.permute(1, 2, 0))
        print('Labels:', target)
        
    def predict_image(self, img):

        xb = img.unsqueeze(0)
        xb = xb.to(self.device)
        yb = self.model(xb)
        _, preds  = torch.max(yb, dim=1)

        self.show_sample(img, target=self.test_dataset.classes[preds[0].item()])

