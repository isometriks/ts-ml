import InputNeuron from "../neuron/input-neuron.ts";
import AttachableLayer from "./attachable-layer.ts";

export default class InputLayer extends AttachableLayer {
  #neurons: InputNeuron[] = []

  constructor(nodes: number) {
    super()

    for (let i=0; i < nodes; i++) {
      this.#neurons.push(new InputNeuron())
    }
  }

  get neurons(): InputNeuron[] {
    return this.#neurons
  }

  setInputs(values: number[]) {
    for (let i = 0; i < values.length; i++) {
      this.neurons[i].value = values[i]
    }
  }
}
