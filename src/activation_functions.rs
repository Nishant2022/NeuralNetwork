use std::f64::consts::E;
use ndarray::Array2;

#[derive(Copy, Clone)]
pub enum ActivationFunctions {
    Sigmoid,
    ReLU,
    Tanh
}

pub struct ActivationFunctionMethods;

impl ActivationFunctionMethods {
    pub fn activate(&self, activation_method: ActivationFunctions, inputs: &Array2<f64>) -> Array2<f64> {
        match activation_method {

            // Activation Code for Sigmoid Function
            ActivationFunctions::Sigmoid => {
                return inputs.map(|x| 1.0 / (1.0 + E.powf(-*x)));
            },

            // Activation Code for ReLU Function
            ActivationFunctions::ReLU => {
                return inputs.map(|x| {
                    if *x > 0.0 {*x}
                        else {0.0}
                });
            },

            // Activation Code for Tanh Function
            ActivationFunctions::Tanh => {
                let positive = inputs.map(|x| E.powf(*x));
                let negative = inputs.map(|x| E.powf(-*x));
                
                return (positive.clone() - negative.clone()) / (positive + negative);
            },
        }
    }

    fn derivative(&self, activation_method: ActivationFunctions, inputs: &Array2<f64>) -> Array2<f64> {
        
        match activation_method {

            // Derivative code for Sigmoid Function
            ActivationFunctions::Sigmoid => {
                let activate_outputs = self.activate(activation_method, &inputs);
                return activate_outputs.clone() * (1.0 - activate_outputs);
            },

            // Derivative code for ReLU Function
            ActivationFunctions::ReLU => {
                return inputs.map(|x| {
                    if *x > 0.0 {1.0}
                    else {0.0}
                })
            },

            // Derivative code for Tanh Function
            ActivationFunctions::Tanh => {
                let activate_outputs: Array2<f64> = self.activate(activation_method, &inputs);
                
                return activate_outputs.map(|x| 1.0 - x.powf(2.0));
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn sigmoid_function_activation() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[0.11920292202211757, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]]);
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::Sigmoid, &inputs), outputs);
    }

    #[test]
    fn sigmoid_function_derivative() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[0.10499358540350653, 0.19661193324148185, 0.25, 0.19661193324148185, 0.10499358540350662]]);
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::Sigmoid, &inputs), outputs);
    }

    #[test]
    fn relu_function_activation() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[0.0, 0.0, 0.0, 1.0, 2.0]]);
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::ReLU, &inputs), outputs);

    }

    #[test]
    fn relu_function_derivative() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[0.0, 0.0, 0.0, 1.0, 1.0]]);
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::ReLU, &inputs), outputs);

    }

    #[test]
    fn tanh_function_activation() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[-0.964027580075817, -0.7615941559557649, 0.0, 0.7615941559557649, 0.964027580075817]]);
        assert_eq!(ActivationFunctionMethods.activate(ActivationFunctions::Tanh, &inputs), outputs);
    }
    
    #[test]
    fn tanh_function_derivative() {
        let inputs: Array2<f64> = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let outputs: Array2<f64> = arr2(&[[0.0706508248531642, 0.41997434161402614, 1.0, 0.41997434161402614, 0.0706508248531642]]);
        assert_eq!(ActivationFunctionMethods.derivative(ActivationFunctions::Tanh, &inputs), outputs);
    }
}