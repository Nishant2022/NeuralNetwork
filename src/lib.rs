use std::usize;

use layer::Layer;
use ndarray::Array2;

mod activation_functions;
mod layer;

use crate::activation_functions::ActivationFunctions;

struct NeuralNetwork {
    input_neuron_num: usize,
    layers: Vec<Layer>,
    input_created: bool,
    output_created: bool,
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork { 
            input_neuron_num: 0, 
            layers: Vec::new(),
            input_created: false,
            output_created: false,
        }
    }

    pub fn add_input(mut self, input_neurons: usize) -> Result<Self, String> {
        
        // If an input layer already exists, return an error
        if self.input_created {
            return Err("You can only have one input layer.".to_string());
        }

        // Create an input layer by changing input_neuron_num from zero to input_neurons
        // and updating input_created to true
        self.input_neuron_num = input_neurons;
        self.input_created = true;
        Ok(self)
    }

    pub fn add_hidden_layer(mut self, neuron_num: usize, activation_function: ActivationFunctions) -> Result<Self, String> {

        // If an input layer was not created or an output layer was created, return an error
        if !self.input_created {
            return Err("Input layer not created.".to_string());
        }
        if self.output_created {
            return Err("Output layer was already created.".to_string());
        }

        // If first hidden layer
        if self.layers.len() == 0 {
            let layer: Layer = Layer::new(self.input_neuron_num, neuron_num, activation_function);
            self.layers.push(layer);
        }
        // Otherwise use last layer
        else {
            let layer: Layer = Layer::new(self.layers[self.layers.len() - 1].get_output_layer_size(), neuron_num, activation_function);
            self.layers.push(layer);
        }
        return Ok(self);
    }

    pub fn add_output_layer(mut self, neuron_num: usize, activation_function: ActivationFunctions) -> Result<Self, String> {
        
        // Check that an output layer was not created already
        if self.output_created {
            return Err("Output layer was already created".to_string());
        }

        // If there are no hidden layer
        if self.layers.len() == 0 {
            let layer: Layer = Layer::new(self.input_neuron_num, neuron_num, activation_function);
            self.layers.push(layer);
        }
        // Otherwise use last the last hidden layer
        else {
            let layer: Layer = Layer::new(self.layers[self.layers.len() - 1].get_output_layer_size(), neuron_num, activation_function);
            self.layers.push(layer);
        }
        
        self.output_created = true;
        return Ok(self);
    }

    fn feed_forward(&self, inputs: Array2<f64>) -> Vec<Array2<f64>> {

        // Outputs will hold each layer's outputs, the first layer being the inputs 
        let mut outputs: Vec<Array2<f64>> = Vec::new();
        outputs.push(inputs);
        for (index, layer) in self.layers.iter().enumerate() {
            // While there is a layer, push that layer's output and move to the next
            outputs.push(layer.activate(&outputs[index]));
        }

        return outputs;
    }
}

#[cfg(test)]
mod test {

    use ndarray::{arr2, Array2};

    use crate::{NeuralNetwork, activation_functions::ActivationFunctions};

    #[test]
    fn neural_network_creation() {

        // Create new, empty neural network with zero input or output neurons
        let nn: NeuralNetwork = NeuralNetwork::new();

        assert_eq!(nn.input_neuron_num, 0);
        assert_eq!(nn.layers.len(), 0);
        assert!(!nn.input_created);
        assert!(!nn.output_created);
    }

    #[test]
    fn neural_network_input_creation_success() {

        // Create new neural network with five input neurons
        let nn: NeuralNetwork = NeuralNetwork::new().add_input(5).unwrap();
        
        assert_eq!(nn.input_neuron_num, 5);
        assert_eq!(nn.layers.len(), 0);
        assert!(nn.input_created);
        assert!(!nn.output_created);
    }

    #[test]
    fn neural_network_input_failure() {

        // Tests that an error is returned if more than one input layer is created
        let mut returns_error_when_incorrect: bool = false;
        let _nn: NeuralNetwork = match NeuralNetwork::new().add_input(5).unwrap().add_input(5) {
            Ok(nn) => nn,
            Err(_) => {
                returns_error_when_incorrect = true;
                NeuralNetwork::new()
            }
        };

        assert!(returns_error_when_incorrect);
    }


    #[test]
    fn hidden_layer_without_input() {

        // Try to add a new hidden layer before the input layer
        let mut returns_error_when_no_input: bool = false;
        let nn: NeuralNetwork = NeuralNetwork::new();
        match nn.add_hidden_layer(5, ActivationFunctions::Sigmoid) {
            Ok(_) => {},
            Err(_) => {
                returns_error_when_no_input = true;
            },
        }

        assert!(returns_error_when_no_input);
    }

    #[test]
    fn hidden_layer_with_output() {
        
        // Try to add a new hidden layer after the output layer
        let mut returns_error_when_existing_output: bool = false;
        let mut nn: NeuralNetwork = NeuralNetwork::new();
        nn.output_created = true;
        match nn.add_hidden_layer(5, ActivationFunctions::Sigmoid) {
            Ok(_) => {},
            Err(_) => {
                returns_error_when_existing_output = true;
            },
        }

        assert!(returns_error_when_existing_output);
    }

    #[test]
    fn neural_network_output_success() {
        
        // Create Neural Netork without a hidden layer
        let nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_output_layer(2, ActivationFunctions::Sigmoid).unwrap();

        assert!(nn.output_created);

        // Create Neural Network with hidden layer
        let nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_hidden_layer(4, ActivationFunctions::Sigmoid).unwrap()
            .add_output_layer(2, ActivationFunctions::Sigmoid).unwrap();

        assert!(nn.output_created);
    }

    #[test]
    fn test_feed_forward_single_input() {
        // Tests feed forward function using single input

        // Create a new neural network
        let mut nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_hidden_layer(3, ActivationFunctions::Sigmoid).unwrap()
            .add_output_layer(2, ActivationFunctions::Sigmoid).unwrap();
        
        // Update weights for test
        let input: Array2<f64> = arr2(&[[-0.5, 0.5]]);
        nn.layers[0].update_weights(arr2(&[[-4.0 / 9.0, -3.0 / 9.0, -2.0 / 9.0], 
                                            [-1.0 / 9.0, 0.0 / 9.0, 1.0 / 9.0], 
                                            [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]])).unwrap();
        nn.layers[1].update_weights(arr2(&[[-4.0 / 9.0, -3.0 / 9.0], 
                                            [-2.0 / 9.0, -1.0 / 9.0], 
                                            [1.0 / 9.0, 2.0 / 9.0], 
                                            [3.0 / 9.0, 4.0 / 9.0]])).unwrap();
        
        // Run the feed forward function
        let outputs: Vec<Array2<f64>> = nn.feed_forward(input);

        assert_eq!(outputs[0], arr2(&[[-0.5, 0.5]])); // Input layer
        assert_eq!(outputs[1], arr2(&[[0.4309986674318094, 0.45842951678320015, 0.4861146822539951]])); // Activation of hidden layer
        assert_eq!(outputs[2], arr2(&[[0.41891059998652674, 0.4841808118596575]])); // Activation of output layer
    }

    #[test]
    fn test_feed_forward_multiple_input() {
        // Tests feed forward function using multiple inputs

        // Create a new neural network
        let mut nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_hidden_layer(3, ActivationFunctions::Sigmoid).unwrap()
            .add_output_layer(2, ActivationFunctions::Sigmoid).unwrap();
        
        // Update weights for test
        let input: Array2<f64> = arr2(&[[-0.5, 0.5], [0.25, 1.0], [5.0, -15.0]]);
        nn.layers[0].update_weights(arr2(&[[-4.0 / 9.0, -3.0 / 9.0, -2.0 / 9.0], 
                                            [-1.0 / 9.0, 0.0 / 9.0, 1.0 / 9.0], 
                                            [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]])).unwrap();
        nn.layers[1].update_weights(arr2(&[[-4.0 / 9.0, -3.0 / 9.0], 
                                            [-2.0 / 9.0, -1.0 / 9.0], 
                                            [1.0 / 9.0, 2.0 / 9.0], 
                                            [3.0 / 9.0, 4.0 / 9.0]])).unwrap();
        
        // Run the feed forward function
        let outputs: Vec<Array2<f64>> = nn.feed_forward(input);

        assert_eq!(outputs[0], arr2(&[[-0.5, 0.5], [0.25, 1.0], [5.0, -15.0]])); // Input layer
        assert_eq!(outputs[1], arr2(&[[0.4309986674318094, 0.45842951678320015, 0.4861146822539951],
                                    [0.43782349911420193, 0.5, 0.5621765008857981],
                                    [0.012953727530695873, 0.004804752887159519, 0.0017729545947921433]])); // Activation of hidden layer
        assert_eq!(outputs[2], arr2(&[[0.41891059998652674, 0.4841808118596575],
                                    [0.4258531007240625, 0.4947469295231579],
                                    [0.3902650563308087, 0.41753105797850615]])); // Activation of output layer
    }
}