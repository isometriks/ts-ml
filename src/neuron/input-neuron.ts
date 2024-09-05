import Neuron from "./neuron.ts";
import Sigmoid from "../function/sigmoid.ts";

export default class InputNeuron extends Neuron {
  #value: number

  constructor(value: number = 0) {
    super(Sigmoid.instance(), 0)
    this.#value = value
  }

  set value(value: number) {
    this.#value = value
  }

  get identifier() {
    return `input_${super.identifier}`
  }


  get label() {
    return `Input Neuron: ${this.#value}`
  }

  output(): number {
    return this.#value
  }
}
