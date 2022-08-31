use activation_functions::{ActivationFunction, SigmoidActivationFunction};
use layer::Layer;

mod activation_functions;
mod layer;

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

    pub fn add_hidden_layer(mut self, neuron_num: usize, activation_function: SigmoidActivationFunction) -> Result<Self, String> {

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

    pub fn add_output_layer(mut self, neuron_num: usize, activation_function: SigmoidActivationFunction) -> Result<Self, String> {
        
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

    fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.activate(&mut outputs);
        }

        return outputs;
    }
}

#[cfg(test)]
mod test {

    use crate::NeuralNetwork;
    use crate::activation_functions::SigmoidActivationFunction;

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
        match nn.add_hidden_layer(5, SigmoidActivationFunction) {
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
        match nn.add_hidden_layer(5, SigmoidActivationFunction) {
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
            .add_output_layer(2, SigmoidActivationFunction).unwrap();

        assert!(nn.output_created);

        // Create Neural Network with hidden layer
        let nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_hidden_layer(4, SigmoidActivationFunction).unwrap()
            .add_output_layer(2, SigmoidActivationFunction).unwrap();

        assert!(nn.output_created);
    }

    #[test]
    fn test_feed_forward() {
        let mut nn: NeuralNetwork = NeuralNetwork::new()
            .add_input(2).unwrap()
            .add_hidden_layer(3, SigmoidActivationFunction).unwrap()
            .add_output_layer(2, SigmoidActivationFunction).unwrap();
        
        let input: Vec<f64> = vec![-0.5, 0.5];
        nn.layers[0].update_weights(vec![-4.0 / 9.0, -3.0 / 9.0, -2.0 / 9.0, -1.0 / 9.0, 0.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]).unwrap();
        nn.layers[1].update_weights(vec![-4.0 / 9.0, -3.0 / 9.0, -2.0 / 9.0, -1.0 / 9.0, 1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]).unwrap();
        assert_eq!(nn.feed_forward(&input), vec![0.3207441922627842, 0.6492657332265155]);
    }
}