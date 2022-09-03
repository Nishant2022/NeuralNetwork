use std::f64::consts::E;

#[derive(Copy, Clone)]
pub enum ActivationFunctions {
    Sigmoid,
    ReLU,
    Tanh
}

pub struct ActivationFunctionMethods;

impl ActivationFunctionMethods {
    pub fn activate(&self, activation_method: ActivationFunctions, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = Vec::new();

        match activation_method {

            // Activation Code for Sigmoid Function
            ActivationFunctions::Sigmoid => {
                for i in 0..inputs.len() {
                    outputs.push(1.0 / (1.0 + E.powf(-inputs[i])));
                }
            },

            // Activation Code for ReLU Function
            ActivationFunctions::ReLU => {
                for i in 0..inputs.len() {
                    outputs.push({
                        if inputs[i] > 0.0 {inputs[i]}
                        else {0.0}
                    })
                }
            },

            // Activation Code for Tanh Function
            ActivationFunctions::Tanh => {
                for i in 0..inputs.len() {
                    let positive: f64 = E.powf(inputs[i]);
                    let negative: f64 = E.powf(-inputs[i]);

                    outputs.push((positive - negative) / (positive + negative));
                }
            },
        }
        return outputs;
    }

    fn derivative(&self, activation_method: ActivationFunctions, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = Vec::new();
        
        match activation_method {

            // Derivative code for Sigmoid Function
            ActivationFunctions::Sigmoid => {
                let activate_outputs: Vec<f64> = self.activate(activation_method, &inputs);
                for i in 0..inputs.len() {
                    outputs.push(activate_outputs[i] * (1.0 - activate_outputs[i]));
                }
            },

            // Derivative code for ReLU Function
            ActivationFunctions::ReLU => {
                for i in 0..inputs.len() {
                    outputs.push({
                        if inputs[i] > 0.0 {1.0}
                        else {0.0}
                    })
                }
            },
            ActivationFunctions::Tanh => {
                let activate_outputs: Vec<f64> = self.activate(activation_method, &inputs);
                for i in 0..inputs.len() {
                    outputs.push(1.0 - activate_outputs[i] * activate_outputs[i]);
                }
            },
        }
        return outputs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_function_activation() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.11920292202211757, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823];
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::Sigmoid, &inputs), outputs);
    }

    #[test]
    fn sigmoid_function_derivative() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.10499358540350653, 0.19661193324148185, 0.25, 0.19661193324148185, 0.10499358540350662];
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::Sigmoid, &inputs), outputs);
    }

    #[test]
    fn relu_function_activation() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::ReLU, &inputs), outputs);

    }

    #[test]
    fn relu_function_derivative() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::ReLU, &inputs), outputs);

    }

    #[test]
    fn tanh_function_activation() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![-0.964027580075817, -0.7615941559557649, 0.0, 0.7615941559557649, 0.964027580075817];
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::Tanh, &inputs), outputs);
    }
    
    #[test]
    fn tanh_function_derivative() {
        let inputs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let outputs: Vec<f64> = vec![0.0706508248531642, 0.41997434161402614, 1.0, 0.41997434161402614, 0.0706508248531642];
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::Tanh, &inputs), outputs);
    }
}