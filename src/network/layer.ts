import Sigmoid from "../function/sigmoid.ts";
import Neuron from "../neuron/neuron.ts";
import BiasNeuron from "../neuron/bias-neuron.ts";

export default class Layer {
  neurons: ConnectableNeuronInterface[] = []

  constructor(nodes: number, func: FunctionInterface = Sigmoid.instance(), bias: boolean = true) {
    for (let i=0; i < nodes; i++) {
      this.neurons.push(this.createNeuron(func))
    }

    if (bias) {
      this.neurons.push(new BiasNeuron())
    }
  }

  attachNeuron(neuron: ConnectableNeuronInterface) {
    for (const neighbor of this.neurons) {
      neuron.addSynapse(neighbor, Math.random())
    }
  }

  attachLayer(layer: Layer) {
    for (const neuron of layer.neurons) {
      this.attachNeuron(neuron)
    }
  }

  createNeuron(func: FunctionInterface) {
    return new Neuron(func)
  }

  output() {
    return this.neurons.map(neuron => neuron.output())
  }

  unCache() {
    this.neurons.forEach(neuron => neuron.unCache())
  }
}
