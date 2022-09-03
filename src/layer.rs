use rand::prelude::*;

use crate::activation_functions::{ActivationFunctions, ActivationFunctionMethods};

// Neural Network layer. Contains the weights between neurons.
pub struct Layer {
    input_neuron_num: usize,
    output_neuron_num: usize,
    weights: Vec<f64>,
    activation_function: ActivationFunctions,
}

impl Layer {
    pub fn new(input_neuron_num: usize, output_neuron_num: usize, activation_function: ActivationFunctions) -> Layer {
        // Create vector of weights
        let mut weights: Vec<f64> = vec![0.0; (input_neuron_num + 1) * output_neuron_num];

        // Randomly initialize weights
        let epsilon_init: f64 = 6.0_f64.sqrt() / ((input_neuron_num + output_neuron_num) as f64).sqrt();
        let mut rng = rand::thread_rng();

        for i in 0..weights.len() {
            weights[i] = rng.gen::<f64>() * 2.0 * epsilon_init - epsilon_init;
        }

        // Create and return layer
        Layer { 
            input_neuron_num: input_neuron_num.try_into().unwrap(), 
            output_neuron_num: output_neuron_num.try_into().unwrap(), 
            weights,
            activation_function,
        }
    }

    pub fn get_input_layer_size(&self) -> usize {
        return self.input_neuron_num;
    }

    pub fn get_output_layer_size(&self) -> usize {
        return self.output_neuron_num;
    }

    pub fn activate(&self, inputs: &mut Vec<f64>) -> Vec<f64> {
        inputs.insert(0, 1.0);

        let mut outputs: Vec<f64> = vec![0.0; self.output_neuron_num];

        for i in 0..self.output_neuron_num {
            for j in 0..(self.input_neuron_num + 1) {
                outputs[i] += inputs[j] * self.weights[i * (self.input_neuron_num + 1) + j];
            }
        }

        return ActivationFunctionMethods.activate(self.activation_function, &outputs);
    }

    pub fn update_weights(&mut self, weights: Vec<f64>) -> Result<Vec<f64>, String> {
        if weights.len() != (self.input_neuron_num + 1) * self.output_neuron_num {
            return Err(format!("Length of weights vector was not correct. Expected a vec of size {}.", (self.input_neuron_num + 1) * self.output_neuron_num));
        }
        self.weights = weights[..].to_vec();
        return Ok(weights);
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn layer_creation() {
        let layer: Layer = Layer::new(5, 4, ActivationFunctions::Sigmoid);
        assert_eq!(layer.input_neuron_num, 5);
        assert_eq!(layer.output_neuron_num, 4);
        assert_eq!(layer.weights.len(), 24);
    }

    #[test]
    fn layer_ativation() {
        let mut layer: Layer = Layer::new(2, 2, ActivationFunctions::Sigmoid);
        layer.weights = vec![-1.0, -0.5, 0.0, 0.75, 0.5, 0.25];

        let mut inputs = vec![-0.5, 0.5];
        let outputs = layer.activate(&mut inputs);

        assert_eq!(outputs, vec![0.320821300824607, 0.6513548646660542]);
    }

    #[test]
    fn layer_random_initialization() {
        let layer: Layer = Layer::new(4, 2, ActivationFunctions::Sigmoid);

        // Make sure that epsilon was calculated correctly
        for weight in &layer.weights {
            assert!(weight.abs() <= 1.0);
        }

        // Make sure that weights are actually different
        assert_ne!(layer.weights[0], layer.weights[1]);
    }

    #[test]
    fn layer_activation_function() {

        // Checks that given activation functions works as expected
        let layer: Layer = Layer::new(4, 2, ActivationFunctions::Sigmoid);
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.11920292202211757, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823];
        assert_eq!(ActivationFunctionMethods.activate(layer.activation_function, &inputs), outputs);
    }

    #[test]
    fn layer_weights_modification_success() {
        let mut layer: Layer = Layer::new(2, 2, ActivationFunctions::Sigmoid);
        
        // Make sure that weights are not initially equal
        assert_ne!(layer.weights, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Check that weights are equal after update
        layer.update_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(layer.weights, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn layer_weights_modification_failure() {
        let mut layer: Layer = Layer::new(2, 2, ActivationFunctions::Sigmoid);
        
        // Make sure that weights are not initially equal
        assert_ne!(layer.weights, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Check that update_weights returns an error when an improper sized vec is given
        let mut error_thrown: bool = false;
        match layer.update_weights(vec![1.0, 2.0]) {
            Ok(_) => {},
            Err(_) => error_thrown = true,
        };
        assert!(error_thrown);
    }

}
