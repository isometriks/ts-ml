import Neuron, { ActivationFunction } from "../neuron/neuron.ts";
import AttachableLayer from "./attachable-layer.ts";

export default class Layer extends AttachableLayer {
  #neurons: ConnectableNeuronInterface[] = []

  constructor(nodes: number, activationFunction: ActivationFunction = ActivationFunction.Sigmoid, bias: number = 0) {
    super()

    for (let i=0; i < nodes; i++) {
      this.#neurons.push(new Neuron(activationFunction, bias))
    }
  }

  output() {
    return this.#neurons.map(neuron => neuron.output())
  }

  unCache() {
    this.#neurons.forEach(neuron => neuron.unCache())
  }

  get neurons(): ConnectableNeuronInterface[] {
    return this.#neurons
  }
}
