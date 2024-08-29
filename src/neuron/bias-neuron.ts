import Neuron from "./neuron.ts";

export default class BiasNeuron extends Neuron {
  output(): number {
    return 1.0
  }

  get identifier() {
    return `bias_${super.identifier}`
  }

  get label() {
    return "Bias Neuron"
  }
}
