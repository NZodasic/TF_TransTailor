import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
import time
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Pruner:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, 
                 scaling_factors={}, importance_scores={}, pruned_filters=set(),
                 train_losses=[], val_losses=[]):
        """
        Initialize the pruner with a model and datasets
        
        Args:
            model: A TensorFlow Keras model
            train_dataset: TF Dataset for training
            val_dataset: TF Dataset for validation
            test_dataset: TF Dataset for testing
            scaling_factors: Dictionary mapping layer indices to scaling factors
            importance_scores: Dictionary mapping layer indices to importance scores
            pruned_filters: Set of tuples (layer_index, filter_index) that have been pruned
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.scaling_factors = scaling_factors
        self.importance_scores = importance_scores
        self.pruned_filters = pruned_filters
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        # Extract all Conv2D layers from the model
        self.conv_layers = []
        self.conv_layer_indices = {}
        
        # Map keras layer names to our internal indices
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2D):
                self.conv_layers.append(layer)
                self.conv_layer_indices[layer.name] = len(self.conv_layers) - 1

    def init_scaling_factors(self):
        """Initialize scaling factors for each convolutional layer"""
        print("Initializing scaling factors from scratch!")
        self.scaling_factors = {}
        
        for i, layer in enumerate(self.conv_layers):
            out_channels = layer.filters
            # Initialize scaling factors as trainable variables
            self.scaling_factors[i] = tf.Variable(
                tf.ones((1, 1, 1, out_channels)),
                trainable=True,
                name=f'scaling_factor_{i}'
            )
    
    def build_scaled_model(self):
        """
        Build a new model that includes scaling factors after each convolutional layer
        
        Returns:
            A new TF model with scaling factors applied
        """
        inputs = self.model.inputs
        x = inputs[0]
        
        # Track connections between layers
        layer_outputs = {}
        scaled_outputs = {}
        
        # Store layer outputs in a dictionary
        for layer in self.model.layers:
            layer_input_tensors = []
            for node in layer._inbound_nodes:
                for inbound_layer in node.inbound_layers:
                    if inbound_layer.name in scaled_outputs:
                        layer_input_tensors.append(scaled_outputs[inbound_layer.name])
                    elif inbound_layer.name in layer_outputs:
                        layer_input_tensors.append(layer_outputs[inbound_layer.name])
            
            # If this is an input layer or has no incoming connections
            if not layer_input_tensors:
                # If it's the input layer
                if layer.name == self.model.input.name.split(':')[0]:
                    layer_outputs[layer.name] = x
                    scaled_outputs[layer.name] = x
                continue
            
            # For layers with single input
            if len(layer_input_tensors) == 1:
                y = layer(layer_input_tensors[0])
            # For layers with multiple inputs (like Add, Concatenate)
            else:
                y = layer(layer_input_tensors)
                
            layer_outputs[layer.name] = y
            
            # Apply scaling factor for Conv2D layers
            if isinstance(layer, Conv2D) and layer.name in self.conv_layer_indices:
                idx = self.conv_layer_indices[layer.name]
                if idx in self.scaling_factors:
                    scaled_y = y * self.scaling_factors[idx]
                    scaled_outputs[layer.name] = scaled_y
                else:
                    scaled_outputs[layer.name] = y
            else:
                scaled_outputs[layer.name] = y
                
        # Get the final output
        outputs = scaled_outputs[self.model.layers[-1].name]
        scaled_model = Model(inputs=inputs, outputs=outputs)
        
        return scaled_model
    
    def train_scaling_factors(self, num_epochs, learning_rate, momentum):
        """
        Train the scaling factors while keeping the model weights fixed
        
        Args:
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            momentum: Momentum for optimizer
        """
        logger.info("===TRAIN SCALING FACTORS===")
        
        # Get a list of trainable variables (only scaling factors)
        trainable_vars = list(self.scaling_factors.values())
        
        # Create scaled model
        scaled_model = self.build_scaled_model()
        
        # Define optimizer and loss
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Define training step
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = scaled_model(images, training=False)
                loss = loss_fn(labels, predictions)
            
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            return loss
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for images, labels in self.train_dataset:
                loss = train_step(images, labels)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def generate_importance_scores(self):
        """Calculate importance scores for each filter based on scaling factors and gradients"""
        print("===Generate importance scores===")
        self.importance_scores = {}
        
        # Build scaled model
        scaled_model = self.build_scaled_model()
        
        # Define loss function
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Compute importance scores using a batch of training data
        for images, labels in self.train_dataset.take(1):  # Use first batch
            with tf.GradientTape() as tape:
                predictions = scaled_model(images, training=False)
                loss = loss_fn(labels, predictions)
            
            # Calculate gradients w.r.t scaling factors
            grads = tape.gradient(loss, list(self.scaling_factors.values()))
            
            # Calculate importance scores
            for i, (layer_idx, scaling_factor) in enumerate(self.scaling_factors.items()):
                grad = grads[i]
                importance = tf.abs(grad * scaling_factor)
                self.importance_scores[layer_idx] = importance
    
    def find_filters_to_prune(self, prune_percentage=5):
        """
        Find filters to prune based on importance scores
        
        Args:
            prune_percentage: Percentage of filters to prune
            
        Returns:
            Dictionary mapping layer indices to lists of filter indices to prune
        """
        # Collect all importance scores
        all_scores = []
        filter_mapping = []  # To keep track of (layer_idx, filter_idx)
        
        for layer_idx, scores_tensor in self.importance_scores.items():
            scores = scores_tensor.numpy().flatten()
            for filter_idx, score in enumerate(scores):
                all_scores.append(score)
                filter_mapping.append((layer_idx, filter_idx))
        
        # Calculate number of filters to prune
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
        
        # Sort filter indices within each layer
        for layer_idx in filters_to_prune:
            filters_to_prune[layer_idx].sort()
        
        print(f"Total filters: {num_filters}, Pruning: {num_to_prune} filters")
        return filters_to_prune
    
    def prune_and_restructure(self, filters_to_prune):
        """
        Prune filters and restructure the model
        
        Args:
            filters_to_prune: Dictionary mapping layer indices to lists of filter indices to prune
        """
        # Create a new model with pruned filters
        old_model = self.model
        input_shape = old_model.input_shape[1:]
        
        # Create a new model with same architecture
        new_model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=10,
            classifier_activation=None  # No activation - outputs raw logits
        )
        
        # Copy weights from old model to new model
        for i, layer in enumerate(old_model.layers):
            if not isinstance(layer, Conv2D) or self.conv_layer_indices.get(layer.name) not in filters_to_prune:
                if i < len(new_model.layers):
                    if new_model.layers[i].name == layer.name:
                        new_model.layers[i].set_weights(layer.get_weights())
            else:
                # This is a Conv2D layer that needs pruning
                layer_idx = self.conv_layer_indices[layer.name]
                filter_indices = filters_to_prune[layer_idx]
                
                # Get original weights and biases
                original_weights = layer.get_weights()
                weights = original_weights[0]  # Shape: [kernel_h, kernel_w, in_channels, out_channels]
                biases = original_weights[1] if len(original_weights) > 1 else None
                
                # Create mask for filters to keep
                mask = np.ones(weights.shape[3], dtype=bool)
                mask[filter_indices] = False
                
                # Prune filters
                pruned_weights = weights[:, :, :, mask]
                pruned_biases = biases[mask] if biases is not None else None
                
                # Find the new layer in the new model
                new_layer = new_model.layers[i]
                
                # Set pruned weights
                if pruned_biases is not None:
                    new_layer.set_weights([pruned_weights, pruned_biases])
                else:
                    new_layer.set_weights([pruned_weights])
                
                # Record pruned filters
                for filter_idx in filter_indices:
                    self.pruned_filters.add((layer_idx, filter_idx))
        
        # Update the model
        self.model = new_model
    
    def prune_scaling_factors(self, filters_to_prune):
        """
        Update scaling factors after pruning
        
        Args:
            filters_to_prune: Dictionary mapping layer indices to lists of filter indices to prune
        """
        for layer_idx in filters_to_prune:
            filter_indices = filters_to_prune[layer_idx]
            if layer_idx in self.scaling_factors:
                # Get current scaling factor
                current_sf = self.scaling_factors[layer_idx].numpy()
                
                # Create mask for filters to keep
                mask = np.ones(current_sf.shape[3], dtype=bool)
                mask[filter_indices] = False
                
                # Create new scaling factor
                new_sf = current_sf[:, :, :, mask]
                
                # Update scaling factor
                self.scaling_factors[layer_idx] = tf.Variable(
                    new_sf,
                    trainable=True,
                    name=f'scaling_factor_{layer_idx}'
                )
    
    def prune_importance_scores(self, filters_to_prune):
        """
        Update importance scores after pruning
        
        Args:
            filters_to_prune: Dictionary mapping layer indices to lists of filter indices to prune
        """
        for layer_idx in filters_to_prune:
            filter_indices = filters_to_prune[layer_idx]
            if layer_idx in self.importance_scores:
                # Get current importance score
                current_is = self.importance_scores[layer_idx].numpy()
                
                # Create mask for filters to keep
                mask = np.ones(current_is.shape[3], dtype=bool)
                mask[filter_indices] = False
                
                # Create new importance score
                new_is = current_is[:, :, :, mask]
                
                # Update importance score
                self.importance_scores[layer_idx] = tf.Variable(
                    new_is,
                    trainable=False
                )
    
    def importance_aware_fine_tuning(self, num_epochs, learning_rate, momentum):
        """
        Fine-tune the model with importance-aware gradients
        
        Args:
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            momentum: Momentum for optimizer
        """
        print("===Importance Aware Fine Tuning===")
        
        # Define optimizer and loss
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Training function
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                # Forward pass through the model
                predictions = self.model(images, training=True)
                loss = loss_fn(labels, predictions)
            
            # Get trainable variables
            trainable_vars = self.model.trainable_variables
            
            # Compute gradients
            gradients = tape.gradient(loss, trainable_vars)
            
            # Apply importance scores to gradients (for convolutional layers)
            modified_gradients = []
            for i, grad in enumerate(gradients):
                layer = None
                for l in self.model.layers:
                    if any(v.name == trainable_vars[i].name for v in l.trainable_variables):
                        layer = l
                        break
                
                if isinstance(layer, Conv2D) and layer.name in self.conv_layer_indices:
                    layer_idx = self.conv_layer_indices[layer.name]
                    if layer_idx in self.importance_scores:
                        # Only modify weights gradients, not biases
                        if 'kernel' in trainable_vars[i].name:
                            # Extract the importance score
                            importance = self.importance_scores[layer_idx]
                            
                            # Reshape importance to match gradient shape
                            reshaped_importance = tf.reshape(importance, 
                                                            [1, 1, 1, importance.shape[3]])
                            
                            # Tile importance to match gradient shape
                            tiled_importance = tf.tile(reshaped_importance, 
                                                      [grad.shape[0], grad.shape[1], grad.shape[2], 1])
                            
                            # Scale gradient by importance
                            modified_grad = grad * tiled_importance
                            modified_gradients.append(modified_grad)
                        else:
                            modified_gradients.append(grad)
                    else:
                        modified_gradients.append(grad)
                else:
                    modified_gradients.append(grad)
            
            # Apply gradients
            optimizer.apply_gradients(zip(modified_gradients, trainable_vars))
            
            return loss
        
        # Training loop
        epoch_loss = 0
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for images, labels in self.train_dataset:
                loss = train_step(images, labels)
                total_loss += loss
                num_batches += 1
            
            epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def finetune(self, num_epochs, learning_rate, momentum, checkpoint_epoch=0):
        """
        Fine-tune the model normally
        
        Args:
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            momentum: Momentum for optimizer
            checkpoint_epoch: Epoch to resume from
        """
        print("\n===Fine-tune the model===")
        
        # Compile the model
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        
        # Create callback to track losses
        class LossHistory(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                nonlocal self_ref
                self_ref.train_losses.append(logs['loss'])
                self_ref.val_losses.append(logs['val_loss'])
        
        # Need a reference to self inside the callback
        self_ref = self
        
        # Train the model
        self.model.fit(
            self.train_dataset,
            epochs=num_epochs,
            validation_data=self.val_dataset,
            initial_epoch=checkpoint_epoch,
            callbacks=[LossHistory()]
        )
    
    def save_state(self, path):
        """
        Save the pruner's state to a file
        
        Args:
            path: Path to save the state
        """
        # Save model weights
        model_weights_path = path + '.h5'
        self.model.save_weights(model_weights_path)
        
        # Convert TF Variables to numpy arrays for pickling
        serializable_scaling_factors = {}
        for k, v in self.scaling_factors.items():
            serializable_scaling_factors[k] = v.numpy()
        
        serializable_importance_scores = {}
        for k, v in self.importance_scores.items():
            serializable_importance_scores[k] = v.numpy()
        
        # Save the rest of the state
        state = {
            'model_weights_path': model_weights_path,
            'scaling_factors': serializable_scaling_factors,
            'importance_scores': serializable_importance_scores,
            'pruned_filters': self.pruned_filters,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'conv_layer_indices': self.conv_layer_indices
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path):
        """
        Load the pruner's state from a file
        
        Args:
            path: Path to load the state from
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Load model weights
        self.model.load_weights(state['model_weights_path'])
        
        # Restore mappings
        self.conv_layer_indices = state['conv_layer_indices']
        
        # Update conv_layers list
        self.conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                self.conv_layers.append(layer)
        
        # Convert numpy arrays back to TF Variables
        self.scaling_factors = {}
        for k, v in state['scaling_factors'].items():
            self.scaling_factors[k] = tf.Variable(v, trainable=True, name=f'scaling_factor_{k}')
        
        self.importance_scores = {}
        for k, v in state['importance_scores'].items():
            self.importance_scores[k] = tf.Variable(v, trainable=False)
        
        # Load other state
        self.pruned_filters = state['pruned_filters']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']
    
    def plot_losses(self, save_path):
        """
        Plot and save training and validation loss curves
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()