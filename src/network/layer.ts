import Sigmoid from "../function/sigmoid.ts";
import Neuron from "../neuron/neuron.ts";
import AttachableLayer from "./attachable-layer.ts";

export default class Layer extends AttachableLayer {
  #neurons: ConnectableNeuronInterface[] = []

  constructor(nodes: number, func: FunctionInterface = Sigmoid.instance(), bias: number = 3) {
    super()

    for (let i=0; i < nodes; i++) {
      this.#neurons.push(new Neuron(func, bias))
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
