use activation_functions::ActivationFunction;
use layer::Layer;

mod activation_functions;
mod layer;

struct NeuralNetwork {
    input_neuron_num: usize,
    layers: Vec<Box<Layer<dyn ActivationFunction>>>,
    input_created: bool,
    output_created: bool,
}

impl NeuralNetwork {
    fn new() -> NeuralNetwork {
        NeuralNetwork { 
            input_neuron_num: 0, 
            layers: Vec::new(),
            input_created: false,
            output_created: false,
        }
    }

    fn add_input(mut self, input_neurons: usize) -> Result<Self, String> {
        
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

}

#[cfg(test)]
mod test {

    use crate::NeuralNetwork;

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
    fn neural_network_input() {

        // Create new neural network with five input neurons
        let nn: NeuralNetwork = NeuralNetwork::new().add_input(5).unwrap();
        
        assert_eq!(nn.input_neuron_num, 5);
        assert_eq!(nn.layers.len(), 0);
        assert!(nn.input_created);
        assert!(!nn.output_created);

        // Tests that an error is returned if more than one input layer is created
        let mut returns_error_when_incorrect: bool = false;
        let _nn = match NeuralNetwork::new().add_input(5).unwrap().add_input(5) {
            Ok(nn) => nn,
            Err(_) => {
                returns_error_when_incorrect = true;
                NeuralNetwork::new()
            }
        };

        assert!(returns_error_when_incorrect);
    }
}