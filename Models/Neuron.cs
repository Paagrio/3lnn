
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Input { get; set; }
        public List<double> Weights { get; set; }
        public double Output { get; set; }

        public Neuron()
        {
            Input = new List<double>();
            Output = 0.00;
            Weights = new List<double>();
        }
    }
}