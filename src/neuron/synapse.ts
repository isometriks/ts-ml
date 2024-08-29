export default class Synapse {
  #neuron: NeuronInterface
  #weight: number

  constructor(neuron: NeuronInterface, weight: number) {
    this.#neuron = neuron
    this.#weight = weight
  }

  get neuron(): NeuronInterface {
    return this.#neuron
  }

  get weight() {
    return this.#weight
  }

  set weight(weight: number) {
    this.#weight = weight
  }

  adjust(delta: number) {
    this.#weight += delta
  }
}
