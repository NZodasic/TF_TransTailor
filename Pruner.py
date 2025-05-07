import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Pruner:
    def __init__(self, model, train_generator, val_generator, test_generator, 
                 train_losses=None, val_losses=None, scaling_factors=None, 
                 importance_scores=None, pruned_filters=None):
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.train_losses = [] if train_losses is None else train_losses
        self.val_losses = [] if val_losses is None else val_losses
        self.scaling_factors = {} if scaling_factors is None else scaling_factors
        self.importance_scores = {} if importance_scores is None else importance_scores
        self.pruned_filters = set() if pruned_filters is None else pruned_filters
        
        # Get model layers
        self.conv_layers = []
        for i, layer in enumerate(self.model.layers[0].layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.conv_layers.append((i, layer))
    
    def init_scaling_factors(self):
        """Initialize scaling factors for all convolutional layers"""
        print("Initializing scaling factors from scratch!")
        
        for i, layer in self.conv_layers:
            # Create scaling factor for each output channel
            self.scaling_factors[i] = tf.Variable(
                tf.ones([1, 1, 1, layer.filters]), 
                trainable=True,
                name=f'scaling_factor_{i}'
            )
    
    def train_scaling_factors(self, num_epochs, learning_rate, momentum):
        """Train scaling factors while keeping model weights fixed"""
        logger.info("===TRAIN SCALING FACTORS===")
        
        # Freeze model weights
        for layer in self.model.layers:
            layer.trainable = False
            
        # Make scaling factors trainable
        trainable_vars = []
        for i in self.scaling_factors:
            trainable_vars.append(self.scaling_factors[i])
        
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            steps_per_epoch = len(self.train_generator)
            
            for step in range(steps_per_epoch):
                x_batch, y_batch = self.train_generator.next()
                
                with tf.GradientTape() as tape:
                    # Forward pass through the model with scaling factors
                    x = x_batch
                    for i, layer in enumerate(self.model.layers[0].layers):
                        if isinstance(layer, tf.keras.layers.Conv2D) and i in self.scaling_factors:
                            x = layer(x) * self.scaling_factors[i]
                        else:
                            x = layer(x)
                    
                    # Apply the remaining layers (GlobalAveragePooling2D and Dense)
                    for layer in self.model.layers[1:]:
                        x = layer(x)
                    
                    loss = loss_fn(y_batch, x)
                
                # Compute gradients and update scaling factors
                grads = tape.gradient(loss, trainable_vars)
                optimizer.apply_gradients(zip(grads, trainable_vars))
                
                total_loss += loss.numpy()
            
            avg_loss = total_loss / steps_per_epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Unfreeze model weights
        for layer in self.model.layers:
            layer.trainable = True
    
    def generate_importance_scores(self):
        """Generate importance scores for convolutional filters based on scaling factors"""
        print("===Generate importance score===")
        self.importance_scores = {}
        
        # Loss function
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        
        # Compute gradients of loss w.r.t. scaling factors
        x_batch, y_batch = self.train_generator.next()
        
        for i, scaling_factor in self.scaling_factors.items():
            with tf.GradientTape() as tape:
                tape.watch(scaling_factor)
                
                # Forward pass through the model with scaling factors
                x = x_batch
                for j, layer in enumerate(self.model.layers[0].layers):
                    if isinstance(layer, tf.keras.layers.Conv2D) and j in self.scaling_factors:
                        x = layer(x)
                        if j == i:
                            x = x * scaling_factor
                    else:
                        x = layer(x)
                
                # Apply the remaining layers
                for layer in self.model.layers[1:]:
                    x = layer(x)
                
                loss = loss_fn(y_batch, x)
            
            # Compute gradient
            grad = tape.gradient(loss, scaling_factor)
            
            # Importance score = |grad * scaling_factor|
            self.importance_scores[i] = tf.abs(grad * scaling_factor)
    
    def find_filters_to_prune(self, prune_percentage=5):
        """Find filters to prune based on importance scores"""
        # Collect all importance scores across all layers
        all_scores = []
        filter_mapping = []  # To keep track of (layer_idx, filter_idx) for each score
        
        for layer_idx, scores_tensor in self.importance_scores.items():
            for filter_idx in range(scores_tensor.shape[-1]):
                score = scores_tensor[0, 0, 0, filter_idx].numpy()
                all_scores.append(score)
                filter_mapping.append((layer_idx, filter_idx))
        
        # Calculate the threshold for bottom k%
        num_filters = len(all_scores)
        num_to_prune = int(num_filters * (prune_percentage / 100))
        
        if num_to_prune == 0:
            print(f"Warning: {prune_percentage}% of {num_filters} filters is less than 1. Defaulting to 1 filter.")
            num_to_prune = 1
        
        # Get indices of bottom k% scores
        sorted_indices = np.argsort(all_scores)
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
        
        print(f"Total filters to prune: {num_to_prune} out of {num_filters}")
        
        return filters_to_prune
    
    def prune_and_restructure(self, filters_to_prune):
        """Prune filters from the model and restructure the model accordingly"""
        # Create a new model with pruned filters
        input_shape = self.model.input_shape[1:]
        base_model = ResNet50(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        
        # Copy weights from original model to new model, excluding pruned filters
        for i, layer in enumerate(base_model.layers):
            if i in filters_to_prune and isinstance(layer, tf.keras.layers.Conv2D):
                # Get original weights and biases
                original_weights = self.model.layers[0].layers[i].get_weights()
                
                # Create mask to exclude pruned filters
                mask = np.ones(original_weights[0].shape[-1], dtype=bool)
                mask[filters_to_prune[i]] = False
                
                # Apply mask to weights and biases
                new_weights = [original_weights[0][..., mask], original_weights[1][mask]]
                
                # Update the layer's weights
                layer.set_weights(new_weights)
                
                # Add to pruned filters set
                for filter_idx in filters_to_prune[i]:
                    self.pruned_filters.add((i, filter_idx))
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # Copy weights for non-pruned layers
                layer.set_weights(self.model.layers[0].layers[i].get_weights())
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # Copy weights for batch normalization layers
                layer.set_weights(self.model.layers[0].layers[i].get_weights())
        
        # Create a new model with pruned base model
        pruned_model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(10, activation='softmax')
        ])
        
        # Copy weights for the dense layer
        pruned_model.layers[-1].set_weights(self.model.layers[-1].get_weights())
        
        # Compile pruned model with same optimizer as original model
        pruned_model.compile(
            optimizer=self.model.optimizer, 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = pruned_model
    
    def prune_scaling_factors(self, filters_to_prune):
        """Prune scaling factors for the filters that have been pruned"""
        for layer_idx in filters_to_prune:
            if layer_idx in self.scaling_factors:
                # Get filter indices to keep
                filter_indices = list(range(self.scaling_factors[layer_idx].shape[-1]))
                keep_indices = [i for i in filter_indices if i not in filters_to_prune[layer_idx]]
                
                # Create new scaling factor tensor with only the kept filters
                new_scaling_factor = tf.gather(
                    self.scaling_factors[layer_idx], 
                    keep_indices, 
                    axis=-1
                )
                
                # Update scaling factor
                self.scaling_factors[layer_idx] = tf.Variable(
                    new_scaling_factor,
                    trainable=True,
                    name=f'scaling_factor_{layer_idx}'
                )
    
    def prune_importance_scores(self, filters_to_prune):
        """Prune importance scores for the filters that have been pruned"""
        for layer_idx in filters_to_prune:
            if layer_idx in self.importance_scores:
                # Get filter indices to keep
                filter_indices = list(range(self.importance_scores[layer_idx].shape[-1]))
                keep_indices = [i for i in filter_indices if i not in filters_to_prune[layer_idx]]
                
                # Create new importance score tensor with only the kept filters
                new_importance_score = tf.gather(
                    self.importance_scores[layer_idx], 
                    keep_indices, 
                    axis=-1
                )
                
                # Update importance score
                self.importance_scores[layer_idx] = new_importance_score
    
    def importance_aware_fine_tuning(self, num_epochs, learning_rate, momentum):
        """Fine-tune the model with importance-aware gradients"""
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        
        for epoch in range(num_epochs):
            total_loss = 0
            steps_per_epoch = len(self.train_generator)
            
            for step in range(steps_per_epoch):
                x_batch, y_batch = self.train_generator.next()
                
                with tf.GradientTape() as tape:
                    # Forward pass through the model with importance-aware gradients
                    x = x_batch
                    activations = {}
                    
                    # Forward pass through base model
                    for i, layer in enumerate(self.model.layers[0].layers):
                        x = layer(x)
                        if isinstance(layer, tf.keras.layers.Conv2D) and i in self.importance_scores:
                            # Store activation for gradient modification
                            activations[i] = x
                    
                    # Apply remaining layers
                    for layer in self.model.layers[1:]:
                        x = layer(x)
                    
                    loss = loss_fn(y_batch, x)
                
                # Get gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                
                # Modify gradients based on importance scores
                modified_grads = []
                var_idx = 0
                
                for i, layer in enumerate(self.model.layers[0].layers):
                    if isinstance(layer, tf.keras.layers.Conv2D) and i in self.importance_scores:
                        # Scale gradients by importance scores
                        grad_kernel = grads[var_idx]
                        grad_bias = grads[var_idx + 1]
                        
                        # Apply importance scaling to kernel gradients
                        importance = self.importance_scores[i]
                        modified_grad_kernel = grad_kernel * tf.reshape(importance, [1, 1, 1, importance.shape[-1]])
                        
                        modified_grads.append(modified_grad_kernel)
                        modified_grads.append(grad_bias)
                        var_idx += 2
                    else:
                        # Keep gradients as they are for non-convolutional layers
                        for _ in range(len(layer.trainable_variables)):
                            modified_grads.append(grads[var_idx])
                            var_idx += 1
                
                # Apply modified gradients
                optimizer.apply_gradients(zip(modified_grads, self.model.trainable_variables))
                
                total_loss += loss.numpy()
            
            avg_loss = total_loss / steps_per_epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch):
        """Finetune the model after pruning"""
        print(f"\n===Fine-tune the model for {num_epochs} epochs===")
        
        # Create optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Training loop
        for epoch in range(checkpoint_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            history = self.model.fit(
                self.train_generator,
                steps_per_epoch=len(self.train_generator),
                epochs=1,
                verbose=1
            )
            
            train_loss = history.history['loss'][0]
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.model.evaluate(
                self.val_generator,
                steps=len(self.val_generator),
                verbose=0
            )[0]
            
            self.val_losses.append(val_loss)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def save_state(self, path):
        """Save the pruner's state to a file"""
        # Save model
        self.model.save(path)
        
        # Save other state information
        state = {
            'scaling_factors': {k: v.numpy() for k, v in self.scaling_factors.items()},
            'importance_scores': {k: v.numpy() for k, v in self.importance_scores.items()},
            'pruned_filters': self.pruned_filters,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path):
        """Load the pruner's state from a file"""
        # Load model
        self.model = tf.keras.models.load_model(path)
        
        # Load other state information
        with open(path + '.pkl', 'rb') as f:
            state = pickle.load(f)
        
        # Convert numpy arrays back to TensorFlow variables
        self.scaling_factors = {
            k: tf.Variable(v, trainable=True, name=f'scaling_factor_{k}') 
            for k, v in state['scaling_factors'].items()
        }
        
        self.importance_scores = {
            k: tf.constant(v) for k, v in state['importance_scores'].items()
        }
        
        self.pruned_filters = state['pruned_filters']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']
    
    def plot_losses(self, save_path):
        """Plot and save training and validation loss curves"""
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()