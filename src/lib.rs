mod activation_functions;

struct Layer {
    input_neuron_num: usize,
    output_neuron_num: usize,
    weights: Vec<f64>,
}

impl Layer {
    fn new(input_neuron_num: u32, output_neuron_num: u32) -> Layer {
        let weights: Vec<f64> = vec![0.0; ((input_neuron_num + 1) * output_neuron_num).try_into().unwrap()];
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
                println!("{:?}", outputs);
            }
        }

        return outputs;
    }
}

#[cfg(test)]
mod tests {

    use crate::Layer;

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

}
