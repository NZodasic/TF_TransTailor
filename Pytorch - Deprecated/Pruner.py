import torch

import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import itertools
import pickle
import os
import logging

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Pruner:
    def __init__(self, model, train_loader, val_loader, test_loader, device, train_losses=[], val_losses=[], scaling_factors={}, importance_scores={}, pruned_filters=set()):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.scaling_factors = scaling_factors
        self.importance_scores = importance_scores
        self.pruned_filters = pruned_filters
        self.train_losses = train_losses
        self.val_losses = val_losses


    def InitScalingFactors(self):
        print("Init scaling factors from scratch!")
        self.scaling_factors = {}
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, nn.Conv2d):
                self.scaling_factors[i] = nn.Parameter(torch.ones((1, layer.out_channels, 1, 1), requires_grad=True).to(self.device))



    def TrainScalingFactors(self, num_epochs, learning_rate, momentum):
        logger.info("===TRAIN SCALING FACTORS===")
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.scaling_factors.items():
            # self.scaling_factors[name] = torch.tensor(param.data.detach(), requires_grad=True, device=param.device)
            self.scaling_factors[name] = param.detach().clone().requires_grad_(True).to(param.device)


        criterion = torch.nn.CrossEntropyLoss()
        num_layers = len(self.model.features)


        params_to_optimize = itertools.chain(self.scaling_factors[sf] for sf in self.scaling_factors.keys())
        optimizer_alpha = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum)
        logger.info("===TRAIN SCALING FACTORS, SETUP===")

        for epoch in range(num_epochs):
            iter_count = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.shape[0]
                optimizer_alpha.zero_grad()
                outputs = inputs
                # outputs.requires_grad = False

                for i in range(num_layers):
                    if isinstance(self.model.features[i], torch.nn.Conv2d):
                        outputs = self.model.features[i](outputs)
                        outputs = outputs.clone() * self.scaling_factors[i].cuda()
                    else:
                        outputs = self.model.features[i](outputs)

                outputs = torch.flatten(outputs, 1)
                classification_output = self.model.classifier(outputs)
                loss = criterion(classification_output, labels)
                loss.backward()
                optimizer_alpha.step()


    def GenerateImportanceScores(self):
        self.model.eval()
        print("===Generate importance score===")
        self.importance_scores = {}
        num_layers = len(self.model.features)
        criterion = torch.nn.CrossEntropyLoss()

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = inputs
            for i in range(num_layers):
                if isinstance(self.model.features[i], torch.nn.Conv2d):
                    outputs = self.model.features[i](outputs)
                    outputs = outputs * self.scaling_factors[i].cuda()
                else:
                    outputs = self.model.features[i](outputs)

            outputs = torch.flatten(outputs, 1)
            classification_output = self.model.classifier(outputs)
            loss = criterion(classification_output, labels)

        # for i, scaling_factor in self.scaling_factors.items():
        #     first_order_derivative = torch.autograd.grad(loss, scaling_factor, retain_graph=True)[0]
        #     self.importance_scores[i] = torch.abs(first_order_derivative * scaling_factor).detach()

        for i in range(num_layers):
            if isinstance(self.model.features[i], torch.nn.Conv2d):
                first_order_derivative = torch.autograd.grad(loss, self.scaling_factors[i], retain_graph=True)[0]
                self.importance_scores[i] = torch.abs(first_order_derivative * self.scaling_factors[i]).detach()
            else:
                self.importance_scores[i] = torch.ones((1, 1)).to(self.device)


    def FindFiltersToPrune(self, prune_percentage=5):
        # Collect all importance scores across all layers
        all_scores = []
        filter_mapping = []  # To keep track of (layer_idx, filter_idx) for each score

        for layer_idx, scores_tensor in self.importance_scores.items():
            for filter_idx, score in enumerate(scores_tensor[0]):
                all_scores.append(score.item())
                filter_mapping.append((layer_idx, filter_idx))

        # Calculate the threshold for bottom k%
        num_filters = len(all_scores)
        num_to_prune = int(num_filters * (prune_percentage / 100))

        if num_to_prune == 0:
            print(f"Warning: {prune_percentage}% of {num_filters} filters is less than 1. Defaulting to 1 filter.")
            num_to_prune = 1

        # Get indices of bottom k% scores
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
        bottom_k_indices = sorted_indices[:num_to_prune]

        # Group filters by layer
        filters_to_prune = {}
        for idx in bottom_k_indices:
            layer_idx, filter_idx = filter_mapping[idx]
            if layer_idx not in filters_to_prune:
                filters_to_prune[layer_idx] = []
            filters_to_prune[layer_idx].append(filter_idx)

        # Sort filter indices within each layer for consistent pruning
        for layer_idx in filters_to_prune:
            filters_to_prune[layer_idx].sort()

        print(f"Total filters: {num_filters}")

        return filters_to_prune


    def PruneSingleFilter(self, layer_index, filter_index):
        conv_layer = self.model.features[layer_index]
        new_conv = torch.nn.Conv2d(in_channels=conv_layer.in_channels,
                                      out_channels=conv_layer.out_channels,
                                      kernel_size=conv_layer.kernel_size,
                                      stride=conv_layer.stride,
                                      padding=conv_layer.padding,
                                      dilation=conv_layer.dilation,
                                      groups=conv_layer.groups,
                                      bias=conv_layer.bias is not None)
        new_filters = torch.cat((conv_layer.weight.data[:filter_index].detach(), conv_layer.weight.data[filter_index + 1:].detach()))
        new_conv.weight.data = new_filters

        if conv_layer.bias is not None:
                new_biases = torch.cat((conv_layer.bias.data[:filter_index], conv_layer.bias.data[filter_index + 1:]))
                new_conv.bias.data = new_biases
        self.pruned_filters.add((layer_index, filter_index))

        return new_conv


    def RestructureNextLayer(self, layer_index, filter_index):
        next_conv_layer = None
        for layer in self.model.features[layer_index + 1:]:
            if isinstance(layer, torch.nn.Conv2d):
                next_conv_layer = layer
                break

        next_new_conv = torch.nn.Conv2d(in_channels=next_conv_layer.in_channels - 1 if next_conv_layer.in_channels > 1 else next_conv_layer.in_channels,
                                            out_channels=next_conv_layer.out_channels,
                                            kernel_size=next_conv_layer.kernel_size,
                                            stride=next_conv_layer.stride,
                                            padding=next_conv_layer.padding,
                                            dilation=next_conv_layer.dilation,
                                            groups=next_conv_layer.groups,
                                            bias=next_conv_layer.bias is not None)

        next_new_conv.weight.data = next_conv_layer.weight.data[:, :filter_index, :, :].clone()
        next_new_conv.weight.data = torch.cat([next_new_conv.weight.data, next_conv_layer.weight.data[:, filter_index + 1:, :, :]], dim=1)

        if next_conv_layer.bias is not None:
            next_new_conv.bias.data = next_conv_layer.bias.data.clone()

        return next_new_conv


    def PruneAndRestructure(self, filters_to_prune):
        for layer_to_prune in filters_to_prune:
            next_layer_index = layer_to_prune + 1
            for i, layer in enumerate(self.model.features[layer_to_prune + 1:]):
                if isinstance(layer, torch.nn.Conv2d):
                    next_layer_index = layer_to_prune + i + 1
                    break
            for filter_to_prune in filters_to_prune[layer_to_prune][::-1]:
                if isinstance(self.model.features[layer_to_prune], torch.nn.Conv2d):
                    self.model.features[layer_to_prune] = self.PruneSingleFilter(layer_to_prune, filter_to_prune)
                if isinstance(self.model.features[next_layer_index], torch.nn.Conv2d):
                    self.model.features[next_layer_index] = self.RestructureNextLayer(layer_to_prune, filter_to_prune)


    def PruneScalingFactors(self, filters_to_prune):
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.scaling_factors[layer_index][0]) if i not in filter_indexes]

            if selected_filters:
                new_scaling_factor = torch.cat(selected_filters)
                new_scaling_factor = new_scaling_factor.view(1, new_scaling_factor.shape[0], 1, 1)
                # Set requires_grad=True for the new scaling factor
                new_scaling_factor.requires_grad_(True)
                self.scaling_factors[layer_index] = new_scaling_factor.to(self.device)


    def PruneImportanceScore(self, filters_to_prune):
        for layer_index in filters_to_prune:
            filter_indexes = filters_to_prune[layer_index]
            selected_filters = [f.unsqueeze(0) for i, f in enumerate(self.importance_scores[layer_index][0]) if i not in filter_indexes]

            if selected_filters:
                new_importance_score = torch.cat(selected_filters)
                new_importance_score = new_importance_score.view(1, new_importance_score.shape[0], 1, 1)
                self.importance_scores[layer_index] = new_importance_score.to(self.device)


    def ModifyClassifier(self):
        last_conv_layer = None
        for layer in self.model.features:
            if isinstance(layer, nn.Conv2d):
                last_conv_layer = layer

        if last_conv_layer is not None and last_conv_layer.out_channels == 0:
            raise ValueError("Last convolutional layer has no output channels after pruning. "
                         "Consider adjusting your pruning strategy.")

        for inputs, labels in self.train_loader:
            self.model.to(self.device)
            inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move inputs and labels to self.device
            outputs = self.model.features(inputs)
            print(outputs.shape)
            break

        # Update the first linear layer in the classifier
        old_fc_layer = self.model.classifier[0]
        new_input_features = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]

        # Create a new fully connected layer with the updated input features
        new_fc_layer = torch.nn.Linear(in_features=new_input_features, out_features=old_fc_layer.out_features)
        new_fc_layer = new_fc_layer.to(self.device)  # Move the new fully connected layer to self.device

        # Replace the old FC layer with the new one in the classifier
        self.model.classifier[0] = new_fc_layer
        self.model.to(self.device)  # Ensure the entire model is on the correct device


#     def ImportanceAwareFineTuning(self, num_epochs, learning_rate, momentum):
#         self.model.train()
#         optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
#         criterion = nn.CrossEntropyLoss()

#         for epoch in range(num_epochs):
#             total_loss = 0
#             for inputs, labels in self.train_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 optimizer.zero_grad()
#                 outputs = inputs
#                 loss = 0

#                 # for i, layer in enumerate(self.model.features):
#                 #     outputs = layer(outputs)
#                 #     if isinstance(layer, nn.Conv2d):
#                 #         outputs = outputs.clone().requires_grad_(True)
#                 #         outputs.register_hook(lambda grad: grad * self.importance_scores[i].detach().to(grad.device))
#                 #         # outputs *= self.importance_scores[i].detach()
#                 #     elif isinstance(layer, nn.ReLU):
#                 #         # Tạo ReLU layer mới với inplace=False
#                 #         outputs = nn.ReLU(inplace=False)(outputs)
#                 #         outputs = outputs.clone().requires_grad_(True)
#                 #         outputs.register_hook(lambda grad: grad * self.importance_scores[i].detach().to(grad.device))

#                 for i, layer in enumerate(self.model.features):
#                     outputs = layer(outputs)
#                     if isinstance(layer, nn.Conv2d):
#                         outputs = outputs.clone().requires_grad_(True)
#                         outputs.register_hook(lambda grad: grad * self.importance_scores[i].detach().to(grad.device))
#                     elif isinstance(layer, nn.ReLU):
#                         # Tạo ReLU layer mới với inplace=False
#                         outputs = nn.ReLU(inplace=False)(outputs)
#                         outputs = outputs.clone().requires_grad_(True)
#                         outputs.register_hook(lambda grad: grad * self.importance_scores[i].detach().to(grad.device))


# # 2, 5, 7 => importance score, 1, 3, 6 => !importance score
#                 outputs = torch.flatten(outputs, 1)
#                 classification_outputs = self.model.classifier(outputs)
#                 loss += criterion(classification_outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}")

    def ImportanceAwareFineTuning(self, num_epochs, learning_rate, momentum):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        
        epoch_loss = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                outputs = inputs
                grad_hooks = []

                for i, layer in enumerate(self.model.features):
                    if isinstance(layer, nn.ReLU):
                        # Use non-inplace ReLU
                        outputs = nn.functional.relu(outputs)
                    else:
                        outputs = layer(outputs)

                    # Add gradient hook for Conv2d layers
                    if isinstance(layer, nn.Conv2d):
                        def create_hook(i):
                            def hook(grad):
                                # Multiply gradient by importance score
                                return grad * self.importance_scores[i].detach().to(grad.device)
                            return hook

                        # Only register hook if output requires gradient
                        if outputs.requires_grad:
                            outputs.register_hook(create_hook(i))

                # Flatten and classify
                outputs = torch.flatten(outputs, 1)
                classification_outputs = self.model.classifier(outputs)

                # Compute and backpropagate loss
                loss = criterion(classification_outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def Finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        """
        Fine-tune the model to achieve W_s*
        
        Args:
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            momentum (float): Momentum value
            checkpoint_epoch (int): Epoch to resume from
        """
        print("\n===Fine-tune the model to achieve W_s*===")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()

        epoch = checkpoint_epoch
        epoch_loss = 0
        
        for epoch in range(epoch, num_epochs):
            print("Epoch " + str(epoch + 1) + "/" + str(num_epochs))
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Calculate average training loss for the epoch
            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            # Calculate average validation loss for the epoch
            val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # # Save loss plot if path is provided
        # if plot_save_path is not None:
        #     self.PlotLosses(train_losses, val_losses, plot_save_path)

    def SaveState(self, path):
        """
        Save the pruner's state to a file.

        Args:
            path (str): The path to save the state to.
        """
        state = {
            'model': self.model,
            'scaling_factors': self.scaling_factors,
            'importance_scores': self.importance_scores,
            'pruned_filters': self.pruned_filters,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def LoadState(self, path):
        """
        Load the pruner's state from a file.

        Args:
            path (str): The path to load the state from.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.scaling_factors = state['scaling_factors']
        self.importance_scores = state['importance_scores']
        self.pruned_filters = state['pruned_filters']
        # self.train_loader = state['train_loader']
        # self.val_loader = state['val_loader']
        # self.test_loader = state['test_loader']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']

    def PlotLosses(self, train_losses, val_losses, save_path):
        """
        Plot and save training and validation loss curves.

        Args:
            train_losses (list): List of training losses
            val_losses (list): List of validation losses
            save_path (str): Path to save the plot
        """
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
