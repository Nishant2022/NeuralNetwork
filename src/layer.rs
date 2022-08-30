use rand::prelude::*;

use crate::activation_functions::SigmoidActivationFunction;

// Neural Network layer. Contains the weights between neurons.
pub struct Layer {
    input_neuron_num: usize,
    output_neuron_num: usize,
    weights: Vec<f64>,
    activation_function: SigmoidActivationFunction,
}

impl Layer {
    pub fn new(input_neuron_num: usize, output_neuron_num: usize, activation_function: SigmoidActivationFunction) -> Layer {
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

    fn activate(&self, inputs: &mut Vec<f64>) -> Vec<f64> {
        inputs.insert(0, 1.0);

        let mut outputs: Vec<f64> = vec![0.0; self.output_neuron_num];

        for i in 0..self.output_neuron_num {
            for j in 0..(self.input_neuron_num + 1) {
                outputs[i] += inputs[j] * self.weights[i * (self.input_neuron_num + 1) + j];
            }
        }

        return outputs;
    }
}

#[cfg(test)]
mod tests {
    
    use crate::activation_functions::SigmoidActivationFunction;
    use crate::activation_functions::ActivationFunction;

    use super::*;

    #[test]
    fn layer_creation() {
        let layer: Layer = Layer::new(5, 4, SigmoidActivationFunction);
        assert_eq!(layer.input_neuron_num, 5);
        assert_eq!(layer.output_neuron_num, 4);
        assert_eq!(layer.weights.len(), 24);
    }

    #[test]
    fn layer_ativation() {
        let mut layer: Layer = Layer::new(2, 2, SigmoidActivationFunction);
        layer.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut inputs = vec![1.0, 2.0];
        let outputs = layer.activate(&mut inputs);

        assert_eq!(outputs, vec![9.0, 21.0]);
    }

    #[test]
    fn layer_random_initialization() {
        let layer: Layer = Layer::new(4, 2, SigmoidActivationFunction);

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
        let layer: Layer = Layer::new(4, 2, SigmoidActivationFunction);
        let inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs = vec![0.11920292202211757, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823];
        assert_eq!(layer.activation_function.activate(&inputs), outputs);
    }

}
