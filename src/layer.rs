use rand::prelude::*;

// Neural Network layer. Contains the weights between neurons.
pub struct Layer {
    input_neuron_num: usize,
    output_neuron_num: usize,
    weights: Vec<f64>,
}

impl Layer {
    fn new(input_neuron_num: usize, output_neuron_num: usize) -> Layer {
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
        }
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
    
    use super::*;

    #[test]
    fn layer_creation() {
        let layer: Layer = Layer::new(5, 4);
        assert_eq!(layer.input_neuron_num, 5);
        assert_eq!(layer.output_neuron_num, 4);
        assert_eq!(layer.weights.len(), 24);
    }

    #[test]
    fn layer_ativation() {
        let mut layer = Layer::new(2, 2);
        layer.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut inputs = vec![1.0, 2.0];
        let outputs = layer.activate(&mut inputs);

        assert_eq!(outputs, vec![9.0, 21.0]);
    }

    #[test]
    fn layer_random_initialization() {
        let layer: Layer = Layer::new(4, 2);

        // Make sure that epsilon was calculated correctly
        for weight in &layer.weights {
            assert!(weight.abs() <= 1.0);
        }

        // Make sure that weights are actually different
        assert_ne!(layer.weights[0], layer.weights[1]);
    }

}
