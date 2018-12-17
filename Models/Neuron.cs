
using System.Collections.Generic;

namespace NeuralNetwork
{
  public class Neuron
  {
    public double Bias { get; set; }
    public double[] Inputs { get; set; }
    public double[] Weights { get; set; }
    public double Output { get; set; }
  }
}