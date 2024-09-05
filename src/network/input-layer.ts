import Layer from "./layer.ts";
import InputNeuron from "../neuron/input-neuron.ts";

export default class InputLayer extends Layer {
  get #neurons(): InputNeuron[] {
    return this.#neurons
  }

  setInputs(values: number[]) {
    for (let i = 0; i < values.length; i++) {
      this.#neurons[i].value = values[i]
    }
  }

  createNeuron() {
    return new InputNeuron()
  }
}
