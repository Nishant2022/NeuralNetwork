pub trait ActivationFunction {
    fn activate(&self, inputs: &Vec<f64>) -> Vec<f64>;
    fn derivative(&self, inputs: &Vec<f64>) -> Vec<f64>;
}

pub struct SigmoidActivationFunction;

impl ActivationFunction for SigmoidActivationFunction {
    fn activate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = Vec::new();
        for i in 0..inputs.len() {
            outputs.push(1.0 / (1.0 + std::f64::consts::E.powf(-inputs[i])));
        }
        return outputs;
    }

    fn derivative(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = Vec::new();
        let activate_outputs: Vec<f64> = self.activate(&inputs);
        for i in 0..inputs.len() {
            outputs.push(activate_outputs[i] * (1.0 - activate_outputs[i]));
        }
        return outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_function_activation() {
        let inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs = vec![0.11920292202211757, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823];
        assert_eq!(SigmoidActivationFunction.activate(&inputs), outputs);
    }

    #[test]
    fn sigmoid_function_derivative() {
        let inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs = vec![0.10499358540350653, 0.19661193324148185, 0.25, 0.19661193324148185, 0.10499358540350662];
        assert_eq!(SigmoidActivationFunction.derivative(&inputs), outputs);
    }
}